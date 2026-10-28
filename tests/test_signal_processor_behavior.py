"""
SignalProcessor behaviour on a converted engine log.

Expected values come from reading the same CSV with numpy, so these tests
pin what the processor returns, not how it computes it.
"""

import numpy as np
import pytest

from src.data.signal_processor import SignalProcessor


@pytest.fixture(scope='module')
def csv_columns(engine_csv):
    data = np.loadtxt(engine_csv, delimiter=',', skiprows=1)
    return {'time': data[:, 0], 'RPM': data[:, 1], 'MAP': data[:, 2], 'EGT1': data[:, 3]}


@pytest.fixture
def processor(engine_reader):
    reader, time_col = engine_reader
    sp = SignalProcessor()
    sp.process_data(reader, time_column=time_col)
    return sp


def test_registers_every_channel_but_time(processor):
    assert processor.get_signal_names() == ['RPM', 'MAP', 'EGT1']


@pytest.mark.parametrize('time_range', [None, (5.0, 10.0)])
def test_statistics_match_numpy(processor, csv_columns, time_range):
    t, y = csv_columns['time'], csv_columns['RPM']
    if time_range is not None:
        y = y[(t >= time_range[0]) & (t <= time_range[1])]
    stats = processor.calculate_statistics(['RPM'], time_range=time_range)['RPM']
    assert stats['mean'] == pytest.approx(y.mean())
    assert stats['std'] == pytest.approx(y.std())
    assert stats['min'] == pytest.approx(y.min())
    assert stats['max'] == pytest.approx(y.max())
    assert stats['rms'] == pytest.approx(np.sqrt(np.mean(y ** 2)))
    assert stats['peak_to_peak'] == pytest.approx(y.max() - y.min())
    assert stats['duty_cycle_threshold'] == pytest.approx(y.mean())
    assert stats['duty_cycle'] == pytest.approx(100.0 * np.mean(y > y.mean()), abs=0.5)


def test_get_statistics_uses_cache(processor):
    first = processor.get_statistics('MAP', (1.0, 3.0))
    assert processor.get_statistics('MAP', (1.0, 3.0)) == first
    processor.clear_statistics_cache()
    assert processor.get_statistics('MAP', (1.0, 3.0)) == first


def test_values_at_time(processor, csv_columns):
    i = 300  # t = 3.00 s
    assert processor.get_signal_at_time('RPM', 3.0) == pytest.approx(csv_columns['RPM'][i])
    values = processor.get_signals_at_time(['RPM', 'MAP'], 3.0)
    assert values['MAP'] == pytest.approx(csv_columns['MAP'][i])
    assert processor.nearest_sample_time('RPM', 3.004) == pytest.approx(3.0)


def test_signal_range(processor, csv_columns):
    t = csv_columns['time']
    mask = (t >= 1.0) & (t <= 2.0)
    data = processor.get_signal_range('MAP', 1.0, 2.0)
    np.testing.assert_allclose(data['x_data'], t[mask])
    np.testing.assert_allclose(data['y_data'], csv_columns['MAP'][mask])


def test_display_data_covers_whole_signal(processor, csv_columns):
    data = processor.get_all_signals(['RPM'])['RPM']
    assert data['x_data'][0] == pytest.approx(0.0)
    assert data['x_data'][-1] == pytest.approx(csv_columns['time'][-1])
    assert data['y_data'].max() == pytest.approx(csv_columns['RPM'].max())
    assert data['y_data'].min() == pytest.approx(csv_columns['RPM'].min())


def test_normalization_is_display_only(processor):
    mean_before = processor.calculate_statistics(['RPM'])['RPM']['mean']
    assert processor.apply_normalization(['RPM'], 'peak') == ['RPM']
    assert processor.is_normalized('RPM')
    assert np.abs(processor.get_all_signals(['RPM'])['RPM']['y_data']).max() == pytest.approx(1.0)
    assert processor.calculate_statistics(['RPM'])['RPM']['mean'] == pytest.approx(mean_before)
    processor.remove_normalization(['RPM'])
    assert not processor.is_normalized('RPM')
    assert processor.get_all_signals(['RPM'])['RPM']['y_data'].max() == pytest.approx(2500.0)


def test_segment_filter_joins_segments_and_restores(processor, csv_columns):
    processor.apply_segment_filter([(2.0, 4.0), (10.0, 12.0)], ['RPM'])
    filtered = processor.get_signal_data('RPM')
    assert filtered.get('filtered')
    assert len(filtered['x_data']) == 402
    assert np.all(np.diff(filtered['x_data']) > 0)
    processor.restore_original_data()
    assert not processor.get_signal_data('RPM').get('filtered')
    stats = processor.calculate_statistics(['RPM'])['RPM']
    assert stats['mean'] == pytest.approx(csv_columns['RPM'].mean())
