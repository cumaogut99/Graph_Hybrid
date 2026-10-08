"""Builds and writes a report in a background thread (the GUI stays responsive)."""

import logging
import os

from PyQt5.QtCore import QThread, pyqtSignal as Signal

from .builder import BuildCancelled, ReportBuilder
from .config import FORMAT_DOCX, ReportConfig
from .data_source import ReportContext

logger = logging.getLogger(__name__)


class ReportWorker(QThread):
    progress = Signal(int, str)   # percent, message
    succeeded = Signal(str)       # output path
    failed = Signal(str)          # error message
    cancelled = Signal()

    def __init__(self, config: ReportConfig, context: ReportContext, output_path: str, parent=None):
        super().__init__(parent)
        self.config = config
        self.context = context
        self.output_path = output_path
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            builder = ReportBuilder(self.config, self.context,
                                    progress=self.progress.emit,
                                    is_cancelled=lambda: self._cancel_requested)
            report = builder.build()
            self.progress.emit(95, "")
            # Write to a temporary file first: a failed write keeps an older report intact
            temp_path = self.output_path + ".tmp"
            if self.config.output_format == FORMAT_DOCX:
                from .docx_writer import write_docx
                write_docx(report, temp_path)
            else:
                from .pptx_writer import write_pptx
                write_pptx(report, temp_path)
            os.replace(temp_path, self.output_path)
            self.progress.emit(100, "")
            self.succeeded.emit(self.output_path)
        except BuildCancelled:
            self.cancelled.emit()
        except PermissionError:
            self.failed.emit(f"The file is open in another program or cannot be written:\n{self.output_path}")
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            self.failed.emit(str(e))
        finally:
            temp_path = self.output_path + ".tmp"
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
