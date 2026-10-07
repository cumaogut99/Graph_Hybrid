# type: ignore
"""
DateTime Axis Item for Plot Manager

Custom axis item for displaying Unix timestamps as readable datetime.
"""

import logging
import datetime
import pyqtgraph as pg

logger = logging.getLogger(__name__)

# Times read without a date (14:30:45, 1h2m3s) are seconds since midnight or
# elapsed seconds, never real timestamps (those are > 10 days after 1970):
# show them as a clock time without the 01/01/1970 date
CLOCK_ONLY_LIMIT = 10 * 86400


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
        
    @staticmethod
    def _clock_string(v, spacing):
        """Seconds -> [-][Nd ]HH:MM:SS(.fff)."""
        sign = '-' if v < 0 else ''
        # Round first so 59.6 s becomes the next minute, not ":60"
        v = round(abs(v), 3) if spacing < 1 else round(abs(v))
        days, rest = divmod(v, 86400)
        hours, rest = divmod(rest, 3600)
        minutes, seconds = divmod(rest, 60)
        prefix = f"{sign}{int(days)}d " if days else sign
        if spacing < 1:
            return f"{prefix}{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
        return f"{prefix}{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

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
                
                if abs(v) < CLOCK_ONLY_LIMIT:
                    strings.append(self._clock_string(v, spacing))
                    continue

                # Timestamps come from naive file datetimes stored as if UTC
                # (CsvToMpaiConverter); format in UTC to show the wall-clock
                # time from the file, as the statistics panel does
                dt = datetime.datetime.fromtimestamp(v, tz=datetime.timezone.utc)
                
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

