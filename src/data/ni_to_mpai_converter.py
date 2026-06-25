"""
NI Format → MPAI Converter

Converts NI TDM/TDX and TDMS files to MPAI directory format.
Matches the CsvToMpaiConverter interface for drop-in use in DataLoader.

Supported formats
-----------------
*.tdm / *.tdx  — NI DIAdem (XML header + binary payload)
*.tdms         — NI Technical Data Management Streaming (requires nptdms)
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from PyQt5.QtCore import QObject, pyqtSignal as Signal

logger = logging.getLogger(__name__)

_CHUNK_ROWS = 500_000


class NiToMpaiConverter(QObject):
    """Convert NI TDM/TDX or TDMS file to MPAI directory format."""

    progress            = Signal(str, int)   # message, percentage 0-100
    finished            = Signal(str)         # output_path on success
    error               = Signal(str)         # error description on failure
    statistics_computed = Signal(dict)        # optional — kept for API compat

    def __init__(
        self,
        source_path: str,
        mpai_path:   str,
        settings:    Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.source_path = source_path
        self.mpai_path   = mpai_path
        self.settings    = settings or {}
        self.cancelled   = False
        self._converter  = None           # active reader ref for cancel()
        self.metrics:    Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def cancel(self):
        self.cancelled = True

    # ------------------------------------------------------------------
    def convert(self) -> bool:
        try:
            t0  = time.time()
            ext = os.path.splitext(self.source_path)[1].lower()

            if not os.path.exists(self.source_path):
                raise FileNotFoundError(f"Source file not found: {self.source_path}")

            self.progress.emit("Dosya açılıyor...", 2)
            reader = self._open_reader(ext)
            self._converter = reader

            channel_names = reader.get_channel_names()
            if not channel_names:
                raise ValueError("Dosyada hiç kanal bulunamadı")

            row_count = reader.get_row_count()
            logger.info("[NI→MPAI] %d kanal, %d satır", len(channel_names), row_count)
            self.progress.emit(
                f"{len(channel_names)} kanal, {row_count:,} satır bulundu", 5
            )

            time_col  = self._resolve_time_column(channel_names)
            col_order = self._order_columns(channel_names, time_col)

            # Ensure synthetic 'time' column is registered if there's no time channel
            if not time_col and 'time' not in col_order:
                col_order = ['time'] + col_order

            sampling_freq = self._detect_sample_rate(reader, time_col, row_count)

            self.progress.emit("MPAI formatına yazılıyor...", 10)
            rows_written = self._write_mpai(
                reader, col_order, time_col, row_count, sampling_freq
            )

            self.progress.emit("Tamamlanıyor...", 95)

            elapsed = time.time() - t0
            self.metrics = {
                'rows':          rows_written,
                'columns':       len(col_order),
                'elapsed_sec':   elapsed,
                'sampling_freq': sampling_freq,
            }

            logger.info(
                "[NI→MPAI] Dönüşüm tamam: %d satır, %.1fs",
                rows_written, elapsed,
            )
            self.progress.emit("Dönüşüm tamamlandı!", 100)
            self.finished.emit(self.mpai_path)
            return True

        except Exception as exc:
            logger.exception("[NI→MPAI] Dönüşüm başarısız")
            self.error.emit(str(exc))
            return False

    # ------------------------------------------------------------------
    def _open_reader(self, ext: str):
        if ext in ('.tdm', '.tdx'):
            from src.data.tdm_reader import TdmReader
            return TdmReader(self.source_path)

        if ext == '.tdms':
            from src.data.tdms_reader import TdmsReader
            if not TdmsReader.is_available():
                raise ImportError(
                    "TDMS desteği için nptdms gereklidir.\n"
                    "Kurulum: pip install nptdms"
                )
            return TdmsReader(self.source_path)

        raise ValueError(f"Desteklenmeyen NI formatı: {ext}")

    # ------------------------------------------------------------------
    def _resolve_time_column(self, names: List[str]) -> Optional[str]:
        requested = self.settings.get('time_column', '')
        if requested and requested in names:
            return requested

        candidates = [
            'time', 'Time', 'TIME', 't', 'T',
            'timestamp', 'Timestamp', 'TIMESTAMP',
            'X_Value', 'x_value', 'xval', 'x', 'X',
            'elapsed', 'Elapsed', 't_s', 'Zeit',
        ]
        for c in candidates:
            if c in names:
                logger.info("[NI→MPAI] Zaman kanalı auto-detect: '%s'", c)
                return c

        for n in names:
            lower = n.lower()
            if 'time' in lower or lower in ('t', 'x', 'elapsed'):
                logger.info("[NI→MPAI] Zaman kanalı kısmi eşleşme: '%s'", n)
                return n

        logger.info("[NI→MPAI] Zaman kanalı bulunamadı, sentetik oluşturulacak")
        return None

    # ------------------------------------------------------------------
    def _order_columns(self, names: List[str], time_col: Optional[str]) -> List[str]:
        if time_col and time_col in names:
            return [time_col] + [c for c in names if c != time_col]
        return list(names)

    # ------------------------------------------------------------------
    def _detect_sample_rate(
        self,
        reader,
        time_col: Optional[str],
        row_count: int,
    ) -> float:
        fs = float(self.settings.get('sampling_frequency', 0) or 0)
        if fs > 0:
            return fs

        if time_col:
            try:
                n_preview = min(500, row_count)
                preview   = reader.read_channel(time_col, start=0, count=n_preview)
                if len(preview) > 5:
                    diffs = np.diff(preview)
                    diffs = diffs[diffs > 0]
                    if len(diffs):
                        dt = float(np.median(diffs))
                        if dt > 0:
                            fs = 1.0 / dt
                            logger.info("[NI→MPAI] Tespit edilen Fs = %.2f Hz", fs)
                            return fs
            except Exception as exc:
                logger.warning("[NI→MPAI] Örnekleme hızı tespit edilemedi: %s", exc)

        return 1000.0  # Fallback

    # ------------------------------------------------------------------
    def _write_mpai(
        self,
        reader,
        col_order:     List[str],
        time_col:      Optional[str],
        row_count:     int,
        sampling_freq: float,
    ) -> int:
        from src.data.data_engine import MpaiStreamWriter

        writer = MpaiStreamWriter(self.mpai_path)
        writer.initialize(col_order, sampling_freq, overwrite=True)

        time_step   = 1.0 / max(sampling_freq, 1.0)
        time_offset = 0.0
        rows_done   = 0

        try:
            for chunk_start in range(0, max(row_count, 1), _CHUNK_ROWS):
                if self.cancelled:
                    break

                chunk_len  = min(_CHUNK_ROWS, row_count - chunk_start)
                chunk_data: Dict[str, np.ndarray] = {}

                # Generate synthetic time first if needed
                if not time_col:
                    t_arr = np.linspace(
                        time_offset,
                        time_offset + (chunk_len - 1) * time_step,
                        chunk_len,
                        dtype=np.float64,
                    )
                    time_offset += chunk_len * time_step
                    chunk_data['time'] = t_arr

                for col in col_order:
                    if col == 'time' and not time_col:
                        continue  # already filled above
                    try:
                        arr = reader.read_channel(col, start=chunk_start, count=chunk_len)
                    except Exception as exc:
                        logger.warning(
                            "[NI→MPAI] Kanal '%s' okunamadı, sıfır dolduruldu: %s", col, exc
                        )
                        arr = np.zeros(chunk_len, dtype=np.float64)

                    arr = np.asarray(arr, dtype=np.float64)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    chunk_data[col] = arr

                writer.write_chunk(chunk_data)
                rows_done += chunk_len

                pct = 10 + int((rows_done / max(row_count, 1)) * 83)
                self.progress.emit(
                    f"Yazılıyor... ({rows_done:,} / {row_count:,} satır)", pct
                )

        finally:
            writer.close()

        logger.info("[NI→MPAI] %d satır yazıldı → %s", rows_done, self.mpai_path)
        return rows_done

    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
