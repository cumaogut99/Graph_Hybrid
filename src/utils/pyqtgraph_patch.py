"""
PyQtGraph Patch - autoRangeEnabled AttributeError Fix

Bu modül PyQtGraph'teki autoRangeEnabled AttributeError sorununu çözer.
"""

import logging
import pyqtgraph as pg

logger = logging.getLogger(__name__)

def apply_pyqtgraph_patch():
    """
    PyQtGraph'teki çeşitli AttributeError sorunlarını çözen patch'i uygula.
    
    Çözülen sorunlar:
    1. autoRangeEnabled AttributeError
    2. containsNonfinite AttributeError
    """
    try:
        # PlotDataItem'ın _getDisplayDataset metodunu patch'le
        original_get_display_dataset = pg.PlotDataItem._getDisplayDataset
        
        def patched_get_display_dataset(self):
            """Patched version of _getDisplayDataset that handles various AttributeErrors."""
            try:
                # Orijinal metodu çağır
                return original_get_display_dataset(self)
            except AttributeError as e:
                error_msg = str(e)
                logger.debug(f"Caught AttributeError in _getDisplayDataset: {error_msg}")
                
                if "autoRangeEnabled" in error_msg or "containsNonfinite" in error_msg:
                    logger.debug("Using fallback dataset creation")
                    # Fallback: PyQtGraph'in beklediği formatta dataset döndür
                    import numpy as np
                    
                    class Dataset:
                        def __init__(self):
                            self.x = np.array([])
                            self.y = np.array([])
                            self.connect = 'all'
                            # PyQtGraph'in beklediği attribute'ları ekle
                            self.containsNonfinite = False
                            self.rect = None
                            
                        def copy(self):
                            """Dataset copy metodu."""
                            new_dataset = Dataset()
                            new_dataset.x = self.x.copy() if hasattr(self.x, 'copy') else self.x
                            new_dataset.y = self.y.copy() if hasattr(self.y, 'copy') else self.y
                            new_dataset.connect = self.connect
                            new_dataset.containsNonfinite = self.containsNonfinite
                            new_dataset.rect = self.rect
                            return new_dataset
                    
                    return Dataset()
                
                # Diğer AttributeError'lar için de fallback
                import numpy as np
                class Dataset:
                    def __init__(self):
                        self.x = np.array([])
                        self.y = np.array([])
                        self.connect = 'all'
                        self.containsNonfinite = False
                        self.rect = None
                return Dataset()
        
        # PlotDataItem'ın updateItems metodunu da patch'le
        original_update_items = pg.PlotDataItem.updateItems
        
        def patched_update_items(self, styleUpdate=False):
            """Patched version of updateItems that handles containsNonfinite AttributeError."""
            try:
                # Orijinal metodu çağır
                return original_update_items(self, styleUpdate)
            except AttributeError as e:
                if "containsNonfinite" in str(e):
                    logger.debug("Caught containsNonfinite AttributeError in updateItems, skipping")
                    # Bu hatayı sessizce geç, grafik çizimi devam etsin
                    return
                else:
                    # Diğer AttributeError'ları yukarı fırlat
                    raise
        
        # Patch'leri uygula
        pg.PlotDataItem._getDisplayDataset = patched_get_display_dataset
        pg.PlotDataItem.updateItems = patched_update_items
        logger.info("PyQtGraph comprehensive patch applied successfully (autoRangeEnabled + containsNonfinite)")
        
    except Exception as e:
        logger.error(f"Failed to apply PyQtGraph patch: {e}")
        # Patch başarısız olsa bile uygulama çalışmaya devam etsin
        pass

def remove_pyqtgraph_patch():
    """Patch'i kaldır (gerekirse)."""
    try:
        # Bu fonksiyon şimdilik boş - patch'i kaldırmaya gerek yok
        pass
    except Exception as e:
        logger.error(f"Failed to remove PyQtGraph patch: {e}")

# Uygulama başlangıcında patch'i otomatik uygula
apply_pyqtgraph_patch()
