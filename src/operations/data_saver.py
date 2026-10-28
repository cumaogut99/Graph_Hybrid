"""
Worker that saves a DataFrame to CSV or Excel in a background thread.

Moved from app.py without changes.
"""

import os

from PyQt5.QtCore import QObject, pyqtSignal as Signal


class DataSaver(QObject):
    """Veri kaydetme işlemlerini ayrı bir thread'de yürüten worker."""
    finished = Signal(str)  # Kaydedilen dosya yolunu gönderir
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, df, file_path, file_filter):
        super().__init__()
        self.df = df
        self.file_path = file_path
        self.file_filter = file_filter

    def run(self):
        """Veri kaydetme işlemini başlatır."""
        try:
            self._save_data()
            self.finished.emit(self.file_path)
        except Exception as e:
            self.error.emit(str(e))

    def _save_data(self):
        """Asıl veri kaydetme mantığı."""
        # Dosya formatına göre kaydet
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        if file_ext == '.csv' or 'CSV' in self.file_filter:
            self.df.write_csv(self.file_path)
        elif file_ext == '.xlsx' or 'Excel' in self.file_filter:
            self.df.write_excel(self.file_path)
        else:
            # Varsayılan CSV
            self.df.write_csv(self.file_path)
