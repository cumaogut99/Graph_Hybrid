"""
Converting columns to numbers and reporting values that are not numbers.

Part of CsvToMpaiConverter; moved from src/data/csv_to_mpai_converter.py without changes.
"""

import polars as pl


class NumericColumnsMixin:
    """Converting columns to numbers and reporting values that are not numbers."""

    def _numeric_expr(self, col: str) -> pl.Expr:
        """Text column -> Float64 expression (null where not a number)."""
        text = pl.col(col).str.strip_chars()
        if self.decimal_comma:
            text = text.str.replace_all(",", ".", literal=True)
        return text.cast(pl.Float64, strict=False)

    def _numeric_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert every text column of a batch to Float64 in one parallel pass.

        Values that are not numbers become null (stored as 0) and are counted
        in self.non_numeric_report, so one "N/A" no longer zeroes a whole
        column and the user is told which columns lost values.
        """
        text_cols = [c for c, dt in df.schema.items() if dt == pl.String]
        if not text_cols:
            return df
        converted = df.select([self._numeric_expr(c).alias(c) for c in text_cols])
        replacements = []
        for c in text_cols:
            # Empty cells are already null in the text column
            failed = converted[c].null_count() - df[c].null_count()
            if failed:
                replacements.append(self._resolve_non_numeric(df[c], converted[c], failed))
        if replacements:
            converted = converted.with_columns(replacements)
        return df.with_columns(converted)

    def _to_numeric_series(self, series: pl.Series) -> pl.Series:
        """Single-column variant of _numeric_batch (time column, sampling)."""
        if series.dtype == pl.String:
            numeric = series.to_frame().select(self._numeric_expr(series.name)).to_series()
            failed = numeric.null_count() - series.null_count()
            if failed:
                numeric = self._resolve_non_numeric(series, numeric, failed)
            return numeric
        if series.dtype == pl.Datetime:
            return series.dt.epoch("us").cast(pl.Float64) / 1e6
        if series.dtype == pl.Date:
            return series.dt.epoch("s").cast(pl.Float64)
        if series.dtype == pl.Boolean:
            return series.cast(pl.Float64)
        return series

    def _resolve_non_numeric(self, text: pl.Series, numeric: pl.Series, failed: int) -> pl.Series:
        """
        A text column had values that are not numbers. If most of them are
        dates or times, store them as seconds instead; count what remains
        invalid.
        """
        total = len(text) - text.null_count()
        if failed * 2 >= total:
            parsed = self._parse_time_text(text)
            if parsed is not None:
                numeric = parsed.alias(text.name)
                self._datetime_columns.add(text.name)
                failed = numeric.null_count() - text.null_count()
        if failed:
            self._count_non_numeric(text.name, failed, total)
        return numeric

    def _count_non_numeric(self, col_name: str, failed: int, total: int):
        counts = self.non_numeric_report.setdefault(col_name, [0, 0])
        counts[0] += failed
        counts[1] += total
