"""
TDMS File Reader

Wraps the nptdms library to expose NI Technical Data Management Streaming
files with the same interface as TdmReader.

Install dependency::

    pip install nptdms

"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from nptdms import TdmsFile as _TdmsFile
    HAS_NPTDMS = True
except ImportError:
    HAS_NPTDMS = False
    logger.warning("[TDMS] nptdms not installed — TDMS support unavailable. "
                   "Install with: pip install nptdms")


class TdmsReader:
    """
    Read NI TDMS files.

    Usage::

        reader = TdmsReader('data.tdms')
        names  = reader.get_channel_names()       # ['Group/Chan', ...]
        arr    = reader.read_channel('Group/Chan')
        d      = reader.to_dict()
    """

    def __init__(self, tdms_path: str):
        if not HAS_NPTDMS:
            raise ImportError(
                "nptdms is required for TDMS support.\n"
                "Install it with:  pip install nptdms"
            )
        if not os.path.exists(tdms_path):
            raise FileNotFoundError(f"TDMS file not found: {tdms_path}")

        self.tdms_path = tdms_path
        self._channels: Dict[str, dict] = {}
        self._scan()

    # ------------------------------------------------------------------
    def _scan(self):
        """Collect channel metadata without loading data arrays."""
        with _TdmsFile.open(self.tdms_path) as f:
            for group in f.groups():
                gname = group.name
                for ch in group.channels():
                    cname    = ch.name
                    full     = f"{gname}/{cname}"
                    try:
                        n = len(ch)
                    except Exception:
                        n = 0
                    self._channels[full] = {
                        'group':      gname,
                        'channel':    cname,
                        'count':      n,
                        'properties': dict(ch.properties),
                    }

        logger.info("[TDMS] Scanned %d channels from %s",
                    len(self._channels), os.path.basename(self.tdms_path))

    # ------------------------------------------------------------------
    def get_channel_names(self) -> List[str]:
        return list(self._channels.keys())

    def get_group_names(self) -> List[str]:
        return list({info['group'] for info in self._channels.values()})

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
        """Read channel data, optionally a slice [start : start+count]."""
        if name not in self._channels:
            raise KeyError(f"Channel '{name}' not found")

        info  = self._channels[name]
        total = info['count']

        if count is None:
            count = max(0, total - start)
        count = min(count, max(0, total - start))
        if count <= 0:
            return np.array([], dtype=np.float64)

        with _TdmsFile.open(self.tdms_path) as f:
            ch = f[info['group']][info['channel']]
            if start == 0 and count >= total:
                data = ch.data
            else:
                data = ch.read_data(offset=start, length=count)

        if data is None or len(data) == 0:
            return np.array([], dtype=np.float64)

        return np.asarray(data, dtype=np.float64)

    def to_dict(self, max_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Return all channels as {name: ndarray}."""
        result: Dict[str, np.ndarray] = {}
        for name in self._channels:
            try:
                result[name] = self.read_channel(name, count=max_rows)
            except Exception as exc:
                logger.warning("[TDMS] Skipping channel '%s': %s", name, exc)
        return result

    @staticmethod
    def is_available() -> bool:
        return HAS_NPTDMS
