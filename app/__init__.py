"""
App Module - TimeGraphApp Refactored Components

Bu modül, app.py'nin refactor edilmiş bileşenlerini içerir.
Her bileşen ayrı bir dosyada organize edilmiştir.

Modüller:
- main_window.py: Ana pencere sınıfı (TimeGraphApp)
- data_operations.py: Veri yükleme ve kaydetme işlemleri
- project_operations.py: MPAI proje dosyası işlemleri (.mpai)
- layout_operations.py: Layout import/export işlemleri

Kullanım:
    from app.main_window import TimeGraphApp
"""

__version__ = "2.0.0"
__all__ = [
    "TimeGraphApp",
    "DataOperations",
    "ProjectOperations", 
    "LayoutOperations"
]

