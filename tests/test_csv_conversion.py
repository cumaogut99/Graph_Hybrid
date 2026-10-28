"""
CSV -> MPAI conversion through DataLoader, as the app imports a file:
delimiters, decimal commas, text times, non-numeric values, generated
time columns and time units.
"""

import numpy as np
import pytest

from conftest import csv_import_settings


def _convert(qapp, tmp_path, name, text, **settings):
    from src.data.data_loader import DataLoader
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    merged = csv_import_settings(path)
    merged.update(settings)
    loader = DataLoader(merged)
    result = {}
    loader.finished.connect(lambda reader, time_col: result.update(reader=reader, time_col=time_col))
    loader.error.connect(lambda msg: result.update(error=msg))
    loader.run()
    assert 'error' not in result, result.get('error')
    reader = result['reader']
    rows = reader.get_row_count()
    columns = {c: np.asarray(reader.load_column_slice(c, 0, rows)) for c in reader.get_column_names()}
    return loader, result['time_col'], columns


def test_semicolon_and_decimal_comma(qapp, tmp_path):
    _, time_col, cols = _convert(qapp, tmp_path, 'semi.csv', 'time;A;B\n0,0;1,5;2\n0,5;2,5;3\n1,0;3,5;4\n',
                                 delimiter=';', decimal_comma=True)
    assert time_col == 'time'
    np.testing.assert_allclose(cols['time'], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(cols['A'], [1.5, 2.5, 3.5])
    np.testing.assert_allclose(cols['B'], [2.0, 3.0, 4.0])


def test_datetime_text_becomes_epoch_seconds(qapp, tmp_path):
    loader, _, cols = _convert(qapp, tmp_path, 'dt.csv',
                               'time,A\n2024-01-15 14:30:45,1\n2024-01-15 14:30:46,2\n2024-01-15 14:30:47,3\n')
    assert loader._datetime_converted
    np.testing.assert_allclose(cols['time'], [1705329045.0, 1705329046.0, 1705329047.0])


def test_clock_times_run_past_midnight(qapp, tmp_path):
    loader, _, cols = _convert(qapp, tmp_path, 'clock.csv', 'time,A\n23:59:59,1\n00:00:00,2\n00:00:01,3\n')
    assert loader._datetime_converted
    np.testing.assert_allclose(cols['time'], [86399.0, 86400.0, 86401.0])


def test_non_numeric_values_are_zero_and_reported(qapp, tmp_path):
    loader, _, cols = _convert(qapp, tmp_path, 'nn.csv', 'time,A,B\n0,1,x\n1,2,3\n2,abc,4\n')
    np.testing.assert_allclose(cols['A'], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(cols['B'], [0.0, 3.0, 4.0])
    assert loader.non_numeric_report == {'A': [1, 3], 'B': [1, 3]}


def test_generated_time_column(qapp, tmp_path):
    _, time_col, cols = _convert(
        qapp, tmp_path, 'gen.csv', 'A,B\n1,2\n3,4\n5,6\n',
        create_custom_time=True, time_mode='Yeni Zaman Kolonu Oluştur', sampling_frequency=10.0,
        start_time_mode='Sıfır', custom_start_time=None, time_unit='saniye',
        new_time_column_name='time_generated')
    assert time_col == 'time_generated'
    np.testing.assert_allclose(cols['time_generated'], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(cols['A'], [1.0, 3.0, 5.0])


def test_millisecond_time_unit_is_scaled_to_seconds(qapp, tmp_path):
    _, _, cols = _convert(qapp, tmp_path, 'ms.csv', 'time,A\n0,1\n500,2\n1000,3\n', time_unit='milisaniye')
    np.testing.assert_allclose(cols['time'], [0.0, 0.5, 1.0])


@pytest.mark.parametrize('rows', [2500])
def test_quoted_rows_are_unwrapped(qapp, tmp_path, rows):
    lines = ['"time,A"'] + ['"%d,%d"' % (i, 2 * i) for i in range(rows)]
    _, _, cols = _convert(qapp, tmp_path, 'quoted.csv', '\n'.join(lines) + '\n')
    np.testing.assert_allclose(cols['A'][:3], [0.0, 2.0, 4.0])
    assert cols['A'].size == rows
