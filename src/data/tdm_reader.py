"""
TDM/TDX File Reader

Reads NI DIAdem TDM/TDX file pairs (XML header + binary data).
No external dependencies required.

TDM format:
  .tdm  — XML header: channel groups, channel metadata, block references
  .tdx  — Binary payload: raw float/int arrays, packed back-to-back

The reader targets the National Instruments USI (Universal Storage Interface)
TDM layout produced by DIAdem / LabVIEW, where the reference chain is:

    tdm_channelgroup --(channels xpointer)--> tdm_channel
    tdm_channel      --(local_columns xpointer)--> localcolumn
    localcolumn      --(values xpointer)--> <type>_sequence
    <type>_sequence  --(values ref)--> block_bdf   (binary descriptor)
    block_bdf        --> byteOffset / length / valueType in the .tdx payload

References appear either as `ref="usi5"` attributes or as XPointer text such as
`#xpointer(id("usi5") id("usi6"))`. Parsing is namespace-agnostic so it works
across TDM variants. Older/flat TDM variants are handled by fallback scans.
"""

import os
import re
import logging
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DTYPE_MAP: Dict[str, type] = {
    'eFloat64Usi': np.float64,
    'eFloat32Usi': np.float32,
    'eInt64Usi':   np.int64,
    'eInt32Usi':   np.int32,
    'eInt16Usi':   np.int16,
    'eInt8Usi':    np.int8,
    'eUInt64Usi':  np.uint64,
    'eUInt32Usi':  np.uint32,
    'eUInt16Usi':  np.uint16,
    'eUInt8Usi':   np.uint8,
    'eStringUsi':  np.object_,
}

_XPOINTER_ID_RE = re.compile(r'id\(\s*["\']([^"\']+)["\']\s*\)')


def _local(tag: str) -> str:
    """Strip namespace URI and return just the local element name."""
    return tag[tag.index('}') + 1:] if tag.startswith('{') else tag


def _find_all_by_local(root: ET.Element, local_name: str) -> List[ET.Element]:
    """Find all descendants whose local name equals *local_name* (namespace-agnostic)."""
    return [el for el in root.iter() if _local(el.tag) == local_name]


def _child_text(el: ET.Element, local_name: str, default: str = '') -> str:
    """Return the text of the first direct child whose local name matches."""
    for child in el:
        if _local(child.tag) == local_name:
            return (child.text or '').strip()
    return default


def _child_el(el: ET.Element, local_name: str) -> Optional[ET.Element]:
    for child in el:
        if _local(child.tag) == local_name:
            return child
    return None


def _attr_or_child(el: ET.Element, names: List[str], default: str = '') -> str:
    """Look up a value first as an attribute, then as a direct child element text."""
    for n in names:
        v = el.get(n)
        if v is not None and v.strip() != '':
            return v.strip()
    for n in names:
        for child in el:
            if _local(child.tag) == n and (child.text or '').strip():
                return child.text.strip()
    return default


def _int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _ids_from_text(text: Optional[str]) -> List[str]:
    """Extract referenced element ids from XPointer text or a bare id list."""
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    ids = _XPOINTER_ID_RE.findall(text)
    if ids:
        return ids
    return [tok.lstrip('#').strip() for tok in re.split(r'[\s,]+', text) if tok.strip()]


def _ref_ids(el: ET.Element, local_name: str) -> List[str]:
    """
    Return the ids referenced by a direct child *local_name* of *el*.

    Handles both attribute references (`ref`, `ref_id`, `idref`) and XPointer
    text content (`#xpointer(id("usi5") id("usi6"))`).
    """
    child = _child_el(el, local_name)
    if child is None:
        return []
    for attr in ('ref', 'ref_id', 'idref'):
        v = child.get(attr, '').strip()
        if v:
            return [v.lstrip('#')]
    return _ids_from_text(child.text)


class TdmReader:
    """
    Read NI TDM/TDX file pairs.

    Uses namespace-agnostic XML parsing so it works with any TDM variant
    (DIAdem, NI LabVIEW export, third-party writers, etc.).

    Usage::

        reader = TdmReader('data.tdm')   # or 'data.tdx'
        names  = reader.get_channel_names()
        arr    = reader.read_channel('Time')
        d      = reader.to_dict()
    """

    def __init__(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.tdx':
            tdm = os.path.splitext(path)[0] + '.tdm'
            if not os.path.exists(tdm):
                raise FileNotFoundError(f"TDM header not found: {tdm}")
            self.tdm_path = tdm
            self.tdx_path = path
        else:
            self.tdm_path = path
            self.tdx_path = None  # resolved from XML

        self._channels: Dict[str, dict] = {}
        self._groups:   Dict[str, List[str]] = {}
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self):
        try:
            tree = ET.parse(self.tdm_path)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid TDM XML: {exc}") from exc

        root = tree.getroot()

        # ----------------------------------------------------------------
        # Resolve TDX path and global byte order from <include><file .../>
        file_byte_order = 'littleEndian'
        for el in root.iter():
            if _local(el.tag) == 'file':
                file_byte_order = el.get('byteOrder', el.get('byte_order', file_byte_order))
                url = el.get('url', '')
                if url and self.tdx_path is None:
                    candidate = os.path.join(os.path.dirname(self.tdm_path), url)
                    if os.path.exists(candidate):
                        self.tdx_path = candidate

        if self.tdx_path is None:
            self.tdx_path = os.path.splitext(self.tdm_path)[0] + '.tdx'

        if not os.path.exists(self.tdx_path):
            raise FileNotFoundError(f"TDX binary file not found: {self.tdx_path}")

        tdx_size = os.path.getsize(self.tdx_path)

        # ----------------------------------------------------------------
        # block_bdf / block → binary descriptor {id: {offset, length, vtype, border}}
        # (USI stores these as attributes; older variants as child elements)
        block_info: Dict[str, dict] = {}
        for blk in _find_all_by_local(root, 'block_bdf') + _find_all_by_local(root, 'block'):
            bid = blk.get('id', '').strip()
            if not bid:
                continue
            block_info[bid] = {
                'offset': _int(_attr_or_child(blk, ['byteOffset', 'blockOffset', 'byte_offset'], '0')),
                'length': _int(_attr_or_child(blk, ['length'], '0')),
                'vtype':  _attr_or_child(blk, ['valueType', 'value_type'], ''),
                'border': _attr_or_child(blk, ['byteOrder', 'byte_order'], file_byte_order),
            }

        # ----------------------------------------------------------------
        # <type>_sequence elements (double_sequence, long_sequence, ...) →
        # bridge a localcolumn's 'values' to the binary block descriptor.
        seq_block: Dict[str, str] = {}
        for el in root.iter():
            ln = _local(el.tag)
            if not ln.endswith('_sequence'):
                continue
            sid = el.get('id', '').strip()
            if not sid:
                continue
            block_ref = ''
            for rid in _ref_ids(el, 'values'):
                if rid in block_info:
                    block_ref = rid
                    break
                if not block_ref:
                    block_ref = rid
            seq_block[sid] = block_ref

        # ----------------------------------------------------------------
        # submatrix → number_of_rows per referenced localcolumn
        lc_rows: Dict[str, int] = {}
        for sm in _find_all_by_local(root, 'submatrix'):
            n = _int(_child_text(sm, 'number_of_rows', '0'))
            if n <= 0:
                continue
            for lc_id in _ref_ids(sm, 'local_columns'):
                lc_rows[lc_id] = n

        # ----------------------------------------------------------------
        # localcolumn → {block, count, vtype}
        lc_info: Dict[str, dict] = {}
        for lc in _find_all_by_local(root, 'localcolumn') + _find_all_by_local(root, 'localColumn'):
            lc_id = lc.get('id', '').strip()
            if not lc_id:
                continue
            count = _int(_child_text(lc, 'number_of_rows', '0')) or lc_rows.get(lc_id, 0)
            vtype = _child_text(lc, 'values_type', '') or _child_text(lc, 'value_type', '')

            block_ref = ''
            for rid in _ref_ids(lc, 'values'):
                if rid in seq_block:           # values → sequence → block
                    block_ref = seq_block[rid]
                    break
                if rid in block_info:          # values → block directly (older variant)
                    block_ref = rid
                    break
            lc_info[lc_id] = {'block': block_ref, 'count': count, 'vtype': vtype}

        # ----------------------------------------------------------------
        # channel groups → name + member channel ids
        group_name_by_id: Dict[str, str] = {}
        chan_to_group: Dict[str, str] = {}
        for cg in _find_all_by_local(root, 'tdm_channelgroup') + _find_all_by_local(root, 'channelGroup'):
            gid = cg.get('id', '').strip()
            gname = _child_text(cg, 'name', '') or 'Group'
            if gid:
                group_name_by_id[gid] = gname
            for cid in _ref_ids(cg, 'channels'):
                chan_to_group[cid] = gname

        # ----------------------------------------------------------------
        # channels → resolve through localcolumn → block
        for ch in _find_all_by_local(root, 'tdm_channel') + _find_all_by_local(root, 'channel'):
            cid = ch.get('id', '').strip()
            cname = _child_text(ch, 'name', '') or cid or 'Channel'
            unit = _child_text(ch, 'unit_string', '') or _child_text(ch, 'unit', '')

            # Resolve group: membership map first, then channel's own <group ref>
            gname = chan_to_group.get(cid, '')
            if not gname:
                for r in _ref_ids(ch, 'group'):
                    if r in group_name_by_id:
                        gname = group_name_by_id[r]
                        break
            if not gname:
                gname = 'Group'

            # local columns: USI xpointer, or old <values ref_id=...> form
            lc_ids = _ref_ids(ch, 'local_columns')
            if not lc_ids:
                lc_ids = _ref_ids(ch, 'values')

            block_ref = ''
            count = 0
            vtype = ''
            for lc_id in lc_ids:
                info = lc_info.get(lc_id)
                if info and info['block']:
                    block_ref = info['block']
                    count = info['count']
                    vtype = info['vtype']
                    break

            if not block_ref or block_ref not in block_info:
                continue

            blk = block_info[block_ref]
            resolved = blk['vtype'] or vtype or 'eFloat64Usi'
            dtype = _DTYPE_MAP.get(resolved, np.float64)

            # length is a value count in USI; trust number_of_rows when present
            if count <= 0:
                count = blk['length']

            # Clamp to what the binary payload can actually provide
            if dtype != np.object_:
                itemsize = np.dtype(dtype).itemsize
                max_count = max(0, (tdx_size - blk['offset']) // itemsize)
                if count <= 0 or count > max_count:
                    count = max_count

            if cname in self._channels:
                cname = f"{gname}/{cname}" if gname else cname

            self._channels[cname] = {
                'group':      gname,
                'dtype':      dtype,
                'offset':     blk['offset'],
                'count':      count,
                'unit':       unit,
                'byte_order': blk['border'],
            }
            self._groups.setdefault(gname, []).append(cname)

        # ----------------------------------------------------------------
        # Fallback: old TDM format where <block> elements ARE the channels
        if not self._channels and block_info:
            logger.warning("[TDM] No channels via USI scan — trying block-as-channel mode")
            self._parse_from_blocks(root, block_info, tdx_size)

        # ----------------------------------------------------------------
        # Prefix channel names with group name when multiple groups exist
        if len(self._groups) > 1:
            new_ch: Dict[str, dict] = {}
            new_gr: Dict[str, List[str]] = {}
            for gname, ch_list in self._groups.items():
                new_gr[gname] = []
                for cname in ch_list:
                    info = self._channels.pop(cname, None)
                    if info is None:
                        continue
                    uname = f"{gname}/{cname}" if not cname.startswith(f"{gname}/") else cname
                    new_ch[uname] = info
                    new_gr[gname].append(uname)
            self._channels = new_ch
            self._groups = new_gr

        if not self._channels:
            logger.error(
                "[TDM] Could not find any channels. Root tag: %s, Child tags: %s",
                _local(root.tag), [_local(c.tag) for c in list(root)[:15]],
            )
            logger.error(
                "[TDM] localColumn ids: %s | block ids: %s | sequence ids: %s",
                list(lc_info.keys())[:10], list(block_info.keys())[:10],
                list(seq_block.keys())[:10],
            )

        logger.info(
            "[TDM] %d channels from %d groups — TDX: %s",
            len(self._channels), len(self._groups),
            os.path.basename(self.tdx_path or ''),
        )

    def _parse_from_blocks(self, root: ET.Element, block_info: Dict[str, dict], tdx_size: int):
        """Old TDM format: <block> elements carry both binary descriptor and channel name."""
        gname = 'Group'
        self._groups.setdefault(gname, [])

        for blk in _find_all_by_local(root, 'block') + _find_all_by_local(root, 'block_bdf'):
            bid = blk.get('id', '').strip()
            if not bid or bid not in block_info:
                continue

            cname = _child_text(blk, 'name', '') or bid
            unit = _child_text(blk, 'unit_string', '') or _child_text(blk, 'unit', '')

            info = block_info[bid]
            dtype = _DTYPE_MAP.get(info['vtype'], np.float64)
            if dtype == np.object_:
                continue
            itemsize = np.dtype(dtype).itemsize

            count = info['length']
            max_count = max(0, (tdx_size - info['offset']) // itemsize)
            if count <= 0 or count > max_count:
                count = max_count
            if count <= 0:
                continue

            self._channels[cname] = {
                'group':      gname,
                'dtype':      dtype,
                'offset':     info['offset'],
                'count':      count,
                'unit':       unit,
                'byte_order': info['border'],
            }
            self._groups[gname].append(cname)

    # ------------------------------------------------------------------
    def get_channel_names(self) -> List[str]:
        return list(self._channels.keys())

    def get_group_names(self) -> List[str]:
        return list(self._groups.keys())

    def get_channel_unit(self, name: str) -> str:
        return self._channels.get(name, {}).get('unit', '')

    def get_row_count(self, channel_name: Optional[str] = None) -> int:
        if channel_name:
            return self._channels.get(channel_name, {}).get('count', 0)
        counts = [c['count'] for c in self._channels.values() if c['count'] > 0]
        return max(counts) if counts else 0

    def read_channel(
        self,
        name: str,
        start: int = 0,
        count: Optional[int] = None,
    ) -> np.ndarray:
        """Read *count* samples from *start* index for channel *name*."""
        if name not in self._channels:
            raise KeyError(
                f"Channel '{name}' not found. "
                f"Available: {list(self._channels.keys())}"
            )

        info   = self._channels[name]
        dtype  = info['dtype']
        total  = info['count']
        offset = info['offset']
        border = info['byte_order']

        if dtype == np.object_:
            logger.warning("[TDM] String channel '%s' not supported.", name)
            return np.array([], dtype=np.float64)

        if count is None:
            count = max(0, total - start)
        count = min(count, max(0, total - start))
        if count <= 0:
            return np.array([], dtype=np.float64)

        itemsize    = np.dtype(dtype).itemsize
        byte_offset = offset + start * itemsize
        order       = '<' if border == 'littleEndian' else '>'
        dt          = np.dtype(dtype).newbyteorder(order)

        with open(self.tdx_path, 'rb') as fh:
            fh.seek(byte_offset)
            raw = fh.read(count * itemsize)

        arr = np.frombuffer(raw, dtype=dt)
        return arr.astype(np.float64, copy=False)

    def to_dict(self, max_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Return all channels as {name: ndarray}."""
        result: Dict[str, np.ndarray] = {}
        for name in self._channels:
            try:
                result[name] = self.read_channel(name, count=max_rows)
            except Exception as exc:
                logger.warning("[TDM] Skipping channel '%s': %s", name, exc)
        return result
