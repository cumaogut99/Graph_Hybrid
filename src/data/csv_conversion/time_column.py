"""
The time column: generating it, parsing text times and scaling time units.

Part of CsvToMpaiConverter; moved from src/data/csv_to_mpai_converter.py without changes.
"""

import logging
from typing import Optional, Dict, Any
import polars as pl
import numpy as np
from src.data.csv_conversion.time_patterns import (
    DATETIME_FORMATS, CLOCK_PATTERN, MIN_SEC_PATTERN, DURATION_PATTERN, TIME_UNIT_FACTORS,
)

logger = logging.getLogger(__name__)


class TimeColumnMixin:
    """The time column: generating it, parsing text times and scaling time units."""

    def _will_generate_time_column(self, schema: Dict[str, Any]) -> bool:
        """Check if a new time column will be added."""
        create_custom = self.settings.get('create_custom_time', False)
        time_col_name = self.settings.get('time_column')
        
        # If explicitly requested OR no valid time column exists
        if create_custom:
            return True
        if not time_col_name or time_col_name not in schema:
            return True
            
        return False

    def _handle_time_column_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle time column creation for batch, maintaining state."""
        create_custom = self.settings.get('create_custom_time', False)
        time_col_name = self.settings.get('time_column')
        new_col_name = self.settings.get('new_time_column_name', 'time_generated')
        
        # DEBUG: Log what we're looking for and what's available
        logger.debug(f"[TIME TRACE] Looking for time_column='{time_col_name}' in columns: {df.columns[:5]}...")
        
        # Scenario A: Generate Custom Time
        # OR Scenario C: Fallback (No time column found)
        time_col_found = time_col_name and time_col_name in df.columns
        
        # If not found by exact name, try to find it by partial match
        if not time_col_found and time_col_name:
            # Try case-insensitive or partial match
            for col in df.columns:
                if col.lower() == time_col_name.lower() or \
                   time_col_name.lower() in col.lower() or \
                   col.lower() in time_col_name.lower():
                    logger.debug(f"[TIME TRACE] Found by partial match: '{time_col_name}' -> '{col}'")
                    time_col_name = col
                    self.settings['time_column'] = col  # Update settings
                    time_col_found = True
                    break
        
        should_generate = create_custom or not time_col_found
        
        logger.debug(f"[TIME TRACE] time_col_found={time_col_found}, should_generate={should_generate}")
        
        if should_generate:
            sampling_freq = self.settings.get('sampling_frequency', 1000.0)
            if sampling_freq <= 0: sampling_freq = 1000.0
            time_step = 1.0 / sampling_freq
            
            # Generate time array for this batch
            n_rows = df.height
            start = self.current_time_offset
            # Linspace is inclusive, arange is not. 
            # We want [start, start + step, ..., start + (n-1)*step]
            time_arr = np.linspace(start, start + (n_rows - 1) * time_step, n_rows, dtype=np.float64)
            
            # Update offset for next batch
            self.current_time_offset += n_rows * time_step
            
            # Add column
            target_name = new_col_name if create_custom else 'time'
            df = df.with_columns(pl.Series(target_name, time_arr))
            logger.debug(f"[TIME TRACE] Generated time column '{target_name}' with {n_rows} rows")
            
        # Scenario B: Use/Fix Existing Time Column
        elif time_col_name in df.columns:
            # Ensure float64
            try:
                col = df[time_col_name]
                logger.debug(f"[TIME TRACE] Using existing time column '{time_col_name}' (dtype={col.dtype})")
                
                # Log first few values for debugging
                if col.len() > 0:
                    sample_values = col.head(min(5, col.len())).to_list()
                    logger.debug(f"[TIME TRACE] Sample values: {sample_values}")
                
                if col.dtype == pl.Utf8 or col.dtype == pl.String:
                    # String column - need special parsing
                    logger.debug(f"[TIME TRACE] Time column is String, attempting conversion...")
                    
                    # Numbers (decimal comma aware) or date/time text -> epoch
                    # seconds; invalid values become 0 and are reported
                    converted_col = self._to_numeric_series(col)
                    if time_col_name in self._datetime_columns:
                        self.time_is_datetime = True
                        logger.debug(f"[TIME TRACE] Parsed as datetime, converted to epoch seconds")
                    df = df.with_columns(converted_col.alias(time_col_name))
                    
                    # Log result
                    result_col = df[time_col_name]
                    if result_col.len() > 0:
                        sample_after = result_col.head(min(5, result_col.len())).to_list()
                        logger.debug(f"[TIME TRACE] After conversion: {sample_after}")
                    
                elif col.dtype in (pl.Datetime, pl.Date):
                    # try_parse_dates gives Datetime/Date; a plain Float64 cast
                    # would yield microseconds, but the time axis expects epoch
                    # seconds (same as the string branch above)
                    logger.debug(f"[TIME TRACE] Converting {col.dtype} to epoch seconds")
                    epoch_s = (col.dt.epoch("us").cast(pl.Float64) / 1e6) if col.dtype == pl.Datetime \
                        else col.dt.epoch("s").cast(pl.Float64)
                    df = df.with_columns(epoch_s.fill_null(0.0).alias(time_col_name))
                    self.time_is_datetime = True
                elif col.dtype not in [pl.Float64, pl.Float32]:
                    # Numeric but not float - simple cast
                    logger.debug(f"[TIME TRACE] Casting {col.dtype} to Float64")
                    df = df.with_columns(col.cast(pl.Float64, strict=False).fill_null(0.0).alias(time_col_name))
                else:
                    # Already float - just fill nulls
                    logger.debug(f"[TIME TRACE] Already Float64, filling nulls")
                    df = df.with_columns(col.fill_null(0.0).alias(time_col_name))

                # Time unit / Unix timestamp chosen in the import dialog
                df = df.with_columns(self._scale_time_column(df[time_col_name]).alias(time_col_name))
                    
            except Exception as e:
                logger.error(f"[TIME TRACE] Failed to process time column: {e}")
                import traceback
                traceback.print_exc()
                
        return df

    def _user_time_format(self, col_name: str) -> Optional[str]:
        """strftime format chosen in the import dialog for the time column."""
        if col_name != self.settings.get('time_column'):
            return None
        fmt = self.settings.get('time_format')
        return fmt if fmt and '%' in fmt else None

    def _parse_time_text(self, text: pl.Series) -> Optional[pl.Series]:
        """
        Read date/time text as seconds (Float64, null where unreadable), or
        None if the column does not hold dates or times.

        Dates with a time become epoch seconds; times without a date become
        seconds since midnight (elapsed seconds for durations). The format
        found in the first batch is tried first in the following ones.
        """
        name = text.name
        text = text.str.strip_chars()
        total = len(text) - text.null_count()
        if total == 0:
            return None

        def readable(result):
            return result is not None and (len(result) - result.null_count()) * 2 >= total

        cached = self._time_parsers.get(name)
        if cached is not None:
            result = self._apply_time_parser(text, name, cached)
            if readable(result):
                return result

        candidates = []
        user_fmt = self._user_time_format(name)
        if user_fmt:
            # The dialog's formats have whole seconds; %.f also accepts a fraction
            candidates.append(('datetime', user_fmt.replace('%S', '%S%.f')))
        candidates.append(('datetime', None))  # Polars infers the format
        candidates += [('datetime', fmt) for fmt in DATETIME_FORMATS]
        candidates += [('clock', None), ('min_sec', None), ('duration', None)]

        for parser in candidates:
            if parser == cached:
                continue
            result = self._apply_time_parser(text, name, parser)
            if readable(result):
                if user_fmt and parser != candidates[0] and name not in self._time_parsers:
                    logger.warning(f"[TIME] '{name}' does not match the chosen format "
                                   f"'{user_fmt}', read as {parser} instead")
                self._time_parsers[name] = parser
                logger.info(f"[TIME] Column '{name}' read as {parser[0]} {parser[1] or ''}")
                return result
        return None

    def _apply_time_parser(self, text: pl.Series, name: str, parser: tuple) -> Optional[pl.Series]:
        kind, fmt = parser
        try:
            if kind == 'datetime':
                dates = text.str.to_datetime(format=fmt, strict=False)
                return (dates.dt.epoch("us").cast(pl.Float64) / 1e6).alias(name)
            if kind == 'clock':
                return self._clock_seconds(text, name)
            if kind == 'min_sec':
                parts = text.str.extract_groups(MIN_SEC_PATTERN)
                minutes = parts.struct.field("1").cast(pl.Float64)
                seconds = parts.struct.field("2").str.replace(",", ".", literal=True).cast(pl.Float64)
                return (minutes * 60 + seconds).alias(name)
            if kind == 'duration':
                parts = text.str.extract_groups(DURATION_PATTERN)
                hours, minutes, seconds = (
                    parts.struct.field(str(i)).str.replace(",", ".", literal=True).cast(pl.Float64)
                    for i in (1, 2, 3))
                frame = pl.DataFrame({"h": hours, "m": minutes, "s": seconds})
                return frame.select(
                    # an empty match ("" or text without any unit) is not a duration
                    pl.when(pl.col("h").is_not_null() | pl.col("m").is_not_null() | pl.col("s").is_not_null())
                    .then(pl.col("h").fill_null(0) * 3600 + pl.col("m").fill_null(0) * 60 + pl.col("s").fill_null(0))
                ).to_series().alias(name)
        except pl.exceptions.PolarsError:
            return None
        return None

    def _clock_seconds(self, text: pl.Series, name: str) -> pl.Series:
        """h:m:s / h/m/s (AM/PM) text -> seconds, continuing past midnight."""
        parts = text.str.extract_groups(CLOCK_PATTERN)
        hours = parts.struct.field("1").cast(pl.Float64).to_numpy()
        minutes = parts.struct.field("2").cast(pl.Float64).to_numpy()
        seconds = parts.struct.field("3").str.replace(",", ".", literal=True).cast(pl.Float64).to_numpy()
        millis = parts.struct.field("4").cast(pl.Float64).fill_null(0).to_numpy()
        ampm = parts.struct.field("5").str.to_lowercase().str.slice(0, 1).fill_null("").to_numpy()

        with np.errstate(invalid='ignore'):
            valid = ~np.isnan(hours) & (minutes < 60) & (seconds < 61)
        is_pm = ampm == 'p'
        has_ampm = (ampm == 'a') | is_pm
        hours = np.where(has_ampm, hours % 12 + np.where(is_pm, 12, 0), hours)
        values = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
        values[~valid] = np.nan

        # A clock that goes back by more than 12 h passed midnight: add a day
        days, last = self._clock_state.get(name, (0, None))
        idx = np.flatnonzero(valid)
        if len(idx):
            seq = values[idx]
            prev = np.concatenate(([seq[0] if last is None else last], seq[:-1]))
            day_steps = np.cumsum(seq < prev - 43200)
            values[idx] = seq + (days + day_steps) * 86400.0
            self._clock_state[name] = (days + int(day_steps[-1]), float(seq[-1]))
        return pl.Series(name, values, dtype=pl.Float64).fill_nan(None)

    def _scale_time_column(self, series: pl.Series) -> pl.Series:
        """Apply the import dialog's time unit / Unix timestamp choice to the time column."""
        if series.name in self._datetime_columns:
            return series  # read from date/time text: already seconds
        factor = TIME_UNIT_FACTORS.get(self.settings.get('time_unit') or 'saniye', 1.0)
        if self.settings.get('time_format') == 'Unix Timestamp':
            if factor == 1.0 and len(series) and float(series.abs().median() or 0) > 1e11:
                factor = 1e-3  # milliseconds since 1970
            self.time_is_datetime = True
        return series * factor if factor != 1.0 else series
