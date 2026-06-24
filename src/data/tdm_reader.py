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


class TdmReader:
    """
    Read NI TDM/TDX file pairs.

    Usage::

        reader = TdmReader('data.tdm')   # or 'data.tdx'
        names  = reader.get_channel_names()
        arr    = reader.read_channel('Time')
        d      = reader.to_dict()          # all channels
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

        # Detect optional namespace URI
        ns_uri = ''
        if root.tag.startswith('{'):
            ns_uri = root.tag[1:root.tag.index('}')]

        def _path(p: str) -> str:
            if ns_uri:
                return p.replace('usi:', f'{{{ns_uri}}}')
            return p.replace('usi:', '')

        def fnd(el, p):
            return el.find(_path(p))

        def fndall(el, p):
            return el.findall(_path(p))

        def txt(el, p, default='') -> str:
            child = fnd(el, p)
            return (child.text or '').strip() if child is not None else default

        def attr(el, name, default='') -> str:
            return (el.get(name) or '').strip()

        # Resolve TDX path from XML <include>
        file_el = fnd(root, './/usi:include/file')
        if file_el is not None:
            url = attr(file_el, 'url')
            if url:
                candidate = os.path.join(os.path.dirname(self.tdm_path), url)
                if os.path.exists(candidate):
                    self.tdx_path = candidate

        if self.tdx_path is None:
            self.tdx_path = os.path.splitext(self.tdm_path)[0] + '.tdx'

        if not os.path.exists(self.tdx_path):
            raise FileNotFoundError(f"TDX binary file not found: {self.tdx_path}")

        # Build id → element lookup for cross-references
        id_map: Dict[str, ET.Element] = {}
        for el in root.iter():
            eid = attr(el, 'id')
            if eid:
                id_map[eid] = el

        # submatrix → number_of_rows per localColumn
        lc_rows: Dict[str, int] = {}
        for sm in fndall(root, './/usi:submatrix'):
            n = int(txt(sm, 'usi:number_of_rows', '0') or '0')
            for lc_id in txt(sm, 'usi:local_columns', '').split():
                lc_rows[lc_id.strip()] = n

        # block → binary descriptor
        block_info: Dict[str, tuple] = {}
        for blk in fndall(root, './/usi:block'):
            bid = attr(blk, 'id')
            if bid:
                offset  = int(txt(blk, 'usi:blockOffset', '0') or '0')
                length  = int(txt(blk, 'usi:length',      '0') or '0')
                vtype   = txt(blk, 'usi:valueType',  'eFloat64Usi')
                border  = txt(blk, 'usi:byteOrder',  'littleEndian')
                block_info[bid] = (offset, length, vtype, border)

        # localColumn → block reference
        lc_info: Dict[str, tuple] = {}
        for lc in fndall(root, './/usi:localColumn'):
            lc_id = attr(lc, 'id')
            if lc_id:
                vtype     = txt(lc, 'usi:values_type', '')
                values_el = fnd(lc, 'usi:values')
                block_ref = attr(values_el, 'ref_id') if values_el is not None else ''
                lc_info[lc_id] = (block_ref, vtype)

        # channel groups → channels
        for cg in fndall(root, './/usi:channelGroup'):
            gname = txt(cg, 'usi:name', 'Group') or 'Group'
            self._groups.setdefault(gname, [])

            for ch in fndall(cg, './/usi:channel'):
                cname = txt(ch, 'usi:name', 'Channel') or 'Channel'
                unit  = txt(ch, 'usi:unit_string', '')

                values_el = fnd(ch, 'usi:values')
                if values_el is None:
                    continue

                lc_ref = attr(values_el, 'ref_id')
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

        # If multiple groups share the same channel name, prefix with group
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

        logger.info(
            "[TDM] %d channels from %d groups — TDX: %s",
            len(self._channels), len(self._groups),
            os.path.basename(self.tdx_path),
        )

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
        """Read *count* samples starting at *start* for channel *name*."""
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
            logger.warning("[TDM] String channel '%s' not supported, returning empty.", name)
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
