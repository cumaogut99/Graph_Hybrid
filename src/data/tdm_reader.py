"""
TDM/TDX File Reader

Reads NI DIAdem TDM/TDX file pairs (XML header + binary data).
No external dependencies required.

TDM format:
  .tdm  — XML header: channel groups, channel metadata, block references
  .tdx  — Binary payload: raw float/int arrays, packed back-to-back
"""

import os
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
}


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
        # Resolve TDX path from <include><file url="..."/></include>
        # Works regardless of namespace
        for el in root.iter():
            if _local(el.tag) == 'file':
                url = el.get('url', '')
                if url:
                    candidate = os.path.join(os.path.dirname(self.tdm_path), url)
                    if os.path.exists(candidate):
                        self.tdx_path = candidate
                break

        if self.tdx_path is None:
            self.tdx_path = os.path.splitext(self.tdm_path)[0] + '.tdx'

        if not os.path.exists(self.tdx_path):
            raise FileNotFoundError(f"TDX binary file not found: {self.tdx_path}")

        # ----------------------------------------------------------------
        # Build id → element map (all elements that carry an 'id' attribute)
        id_map: Dict[str, ET.Element] = {}
        for el in root.iter():
            eid = el.get('id', '').strip()
            if eid:
                id_map[eid] = el

        # ----------------------------------------------------------------
        # submatrix → number_of_rows per localColumn
        lc_rows: Dict[str, int] = {}
        for sm in _find_all_by_local(root, 'submatrix'):
            n_text = _child_text(sm, 'number_of_rows', '0')
            n = int(n_text or '0')
            lc_ids = _child_text(sm, 'local_columns', '')
            for lc_id in lc_ids.split():
                lc_rows[lc_id.strip()] = n

        # ----------------------------------------------------------------
        # block → binary descriptor  (supports both <block> and <block_bdf>)
        block_info: Dict[str, tuple] = {}
        for blk in list(_find_all_by_local(root, 'block')) + list(_find_all_by_local(root, 'block_bdf')):
            bid = blk.get('id', '').strip()
            if bid:
                offset = int(_child_text(blk, 'blockOffset',  '0') or '0')
                length = int(_child_text(blk, 'length',       '0') or '0')
                vtype  =    _child_text(blk, 'valueType',  'eFloat64Usi') or \
                            _child_text(blk, 'value_type', 'eFloat64Usi')
                border =    _child_text(blk, 'byteOrder',  'littleEndian') or \
                            _child_text(blk, 'byte_order', 'littleEndian')
                block_info[bid] = (offset, length, vtype or 'eFloat64Usi',
                                   border or 'littleEndian')

        # ----------------------------------------------------------------
        # localColumn → block reference
        lc_info: Dict[str, tuple] = {}
        for lc in _find_all_by_local(root, 'localColumn'):
            lc_id = lc.get('id', '').strip()
            if lc_id:
                vtype = (_child_text(lc, 'values_type', '') or
                         _child_text(lc, 'value_type', ''))
                values_el = _child_el(lc, 'values')
                block_ref = (values_el.get('ref_id', '').strip()
                             if values_el is not None else '')
                lc_info[lc_id] = (block_ref, vtype)

        # ----------------------------------------------------------------
        # channel groups → channels
        groups_found: Dict[str, List[str]] = {}

        for cg in _find_all_by_local(root, 'channelGroup'):
            gname = _child_text(cg, 'name', 'Group') or 'Group'
            groups_found.setdefault(gname, [])

            for ch in _find_all_by_local(cg, 'channel'):
                cname = _child_text(ch, 'name', 'Channel') or 'Channel'
                unit  = _child_text(ch, 'unit_string', '') or _child_text(ch, 'unit', '')

                values_el = _child_el(ch, 'values')
                if values_el is None:
                    continue

                lc_ref = values_el.get('ref_id', '').strip()
                if not lc_ref or lc_ref not in lc_info:
                    continue

                block_ref, vtype = lc_info[lc_ref]
                n_rows = lc_rows.get(lc_ref, 0)

                if not block_ref or block_ref not in block_info:
                    continue

                offset, length, btype, border = block_info[block_ref]
                resolved = vtype or btype or 'eFloat64Usi'
                dtype = _DTYPE_MAP.get(resolved, np.float64)

                if n_rows == 0 and length > 0 and dtype != np.object_:
                    n_rows = length // np.dtype(dtype).itemsize

                self._channels[cname] = {
                    'group':      gname,
                    'dtype':      dtype,
                    'offset':     offset,
                    'count':      n_rows,
                    'unit':       unit,
                    'byte_order': border,
                }
                groups_found[gname].append(cname)

        self._groups = groups_found

        # ----------------------------------------------------------------
        # If no channels found via channelGroup, try flat <channel> elements
        if not self._channels:
            logger.warning(
                "[TDM] No channels via channelGroup — trying flat scan"
            )
            self._parse_flat(root, id_map, lc_info, lc_rows, block_info)

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
                    uname = f"{gname}/{cname}"
                    new_ch[uname] = info
                    new_gr[gname].append(uname)
            self._channels = new_ch
            self._groups   = new_gr

        if not self._channels:
            # Log XML structure for debugging
            logger.error(
                "[TDM] Could not find any channels. Root tag: %s, "
                "Child tags: %s",
                root.tag,
                [_local(c.tag) for c in list(root)[:15]],
            )
            logger.error(
                "[TDM] All localColumn ids: %s | All block ids: %s",
                list(lc_info.keys())[:10],
                list(block_info.keys())[:10],
            )

        logger.info(
            "[TDM] %d channels from %d groups — TDX: %s",
            len(self._channels), len(self._groups),
            os.path.basename(self.tdx_path or ''),
        )

    def _parse_flat(self, root, id_map, lc_info, lc_rows, block_info):
        """Fallback: find <channel> elements anywhere in the tree."""
        gname = 'Group'
        self._groups.setdefault(gname, [])

        for ch in _find_all_by_local(root, 'channel'):
            cname = _child_text(ch, 'name', '') or ch.get('id', 'Channel')
            unit  = _child_text(ch, 'unit_string', '') or _child_text(ch, 'unit', '')

            values_el = _child_el(ch, 'values')
            if values_el is None:
                continue

            lc_ref = values_el.get('ref_id', '').strip()
            if not lc_ref or lc_ref not in lc_info:
                continue

            block_ref, vtype = lc_info[lc_ref]
            n_rows = lc_rows.get(lc_ref, 0)

            if not block_ref or block_ref not in block_info:
                continue

            offset, length, btype, border = block_info[block_ref]
            resolved = vtype or btype or 'eFloat64Usi'
            dtype = _DTYPE_MAP.get(resolved, np.float64)

            if n_rows == 0 and length > 0 and dtype != np.object_:
                n_rows = length // np.dtype(dtype).itemsize

            self._channels[cname] = {
                'group':      gname,
                'dtype':      dtype,
                'offset':     offset,
                'count':      n_rows,
                'unit':       unit,
                'byte_order': border,
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
