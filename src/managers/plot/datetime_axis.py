# type: ignore
"""
DateTime Axis Item for Plot Manager

Custom axis item for displaying Unix timestamps as readable datetime.
"""

import logging
import datetime
import pyqtgraph as pg

logger = logging.getLogger(__name__)


class DateTimeAxisItem(pg.AxisItem):
    """Custom axis item for displaying Unix timestamps as readable datetime."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_datetime_axis = False
        
    def enable_datetime_mode(self, enable=True):
        """Enable or disable datetime formatting."""
        self.is_datetime_axis = enable
        # Force axis update
        self.picture = None  # Clear cache to force redraw
        self.update()
        
    def tickStrings(self, values, scale, spacing):
        """Override to format Unix timestamps as datetime strings."""
        if not self.is_datetime_axis:
            return super().tickStrings(values, scale, spacing)
            
        strings = []
        for v in values:
            try:
                # ROBUST: Milisaniye timestamp kontrolü (1e12'den büyük)
                if abs(v) > 1e12:
                    # Milisaniye cinsinden, saniyeye çevir
                    v = v / 1000.0
                
                # Convert Unix timestamp to datetime (local time)
                dt = datetime.datetime.fromtimestamp(v)
                
                # Choose format based on time range
                if spacing < 1:  # Less than 1 second - show milliseconds
                    time_str = dt.strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
                elif spacing < 60:  # Less than 1 minute - show seconds
                    time_str = dt.strftime('%d/%m/%Y %H:%M:%S')
                elif spacing < 3600:  # Less than 1 hour - show minutes
                    time_str = dt.strftime('%d/%m %H:%M')
                elif spacing < 86400:  # Less than 1 day - show hours
                    time_str = dt.strftime('%d/%m %H:%M')
                else:  # More than 1 day - show date
                    time_str = dt.strftime('%d/%m/%Y')
                    
                strings.append(time_str)
            except (ValueError, OSError, OverflowError) as e:
                # Fallback to original formatting if timestamp is invalid
                logger.debug(f"Datetime formatting failed for value {v}: {e}")
                strings.append(f'{v:.2f}')
                
        return strings

