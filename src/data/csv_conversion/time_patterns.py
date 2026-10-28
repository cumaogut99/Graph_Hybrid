"""
Time formats and patterns the converter recognizes in text time columns.

Moved from src/data/csv_to_mpai_converter.py without changes.
"""

# Date + time formats tried when Polars cannot infer the format itself.
# %.f makes fractional seconds optional.
DATETIME_FORMATS = (
    "%m/%d/%Y %I:%M:%S%.f %p", "%d/%m/%Y %I:%M:%S%.f %p", "%Y-%m-%d %I:%M:%S%.f %p",
    "%d.%m.%Y %H:%M:%S%.f", "%d/%m/%Y %H:%M:%S%.f", "%m/%d/%Y %H:%M:%S%.f",
    "%Y/%m/%d %H:%M:%S%.f", "%d-%m-%Y %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f",
    "%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M",
)

# Times without a date. Seconds may have a fraction (. or ,).
#   h:m:s, h/m/s, h:m:s:ms (optionally AM/PM)  -> clock time (or elapsed hours)
#   m:s                                         -> minutes and seconds
#   1h2m3.5s, 2m 30s, 45s                       -> duration
_SECONDS = r"(\d{1,2}(?:[.,]\d+)?)"
CLOCK_PATTERN = (r"^(\d{1,3})[:/](\d{1,2})[:/]" + _SECONDS
                 + r"(?::(\d{1,3}))?\s*([AaPp]\.?[Mm]\.?)?$")
MIN_SEC_PATTERN = r"^(\d{1,4}):" + _SECONDS + r"$"
DURATION_PATTERN = (r"(?i)^(?:(\d+(?:[.,]\d+)?)\s*h(?:ours?|rs?)?)?\s*"
                    r"(?:(\d+(?:[.,]\d+)?)\s*m(?:in(?:utes?)?)?)?\s*"
                    r"(?:(\d+(?:[.,]\d+)?)\s*s(?:ec(?:onds?)?)?)?$")

# "Zaman Birimi" of the import dialog -> factor to seconds
TIME_UNIT_FACTORS = {'saniye': 1.0, 'milisaniye': 1e-3, 'mikrosaniye': 1e-6, 'nanosaniye': 1e-9}
