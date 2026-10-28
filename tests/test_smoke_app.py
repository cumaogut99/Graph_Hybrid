"""
End-to-end smoke test: open the main window without a display, import a
CSV through the import dialog and the loader thread, plot channels and
use the main views. Catches wiring errors that unit tests miss.
"""

import importlib.util
import time

import pytest

from conftest import ROOT


def _wait(qapp, condition, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def _settle(qapp, seconds=0.5):
    _wait(qapp, lambda: False, seconds)


@pytest.fixture
def main_window(qapp):
    spec = importlib.util.spec_from_file_location('app_main', ROOT / 'app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    window = module.TimeGraphApp()
    yield window
    window.close()
    _settle(qapp, 0.2)


def test_import_plot_and_analyse(qapp, main_window, engine_csv):
    from src.data.data_import_dialog import DataImportDialog

    dialog = DataImportDialog(str(engine_csv))
    assert _wait(qapp, lambda: dialog.preview_df is not None, 10), 'preview did not load'
    settings = dialog.get_import_settings()
    assert settings['time_column'] == 'time'
    dialog.close()

    main_window.data_ops.load_data_with_settings(settings)
    assert _wait(qapp, lambda: len(main_window.file_manager.loaded_files) == 1), 'file did not load'
    _settle(qapp)

    widget = main_window.widget_container_manager.get_active_widget()
    processor = widget.signal_processor
    assert processor.get_signal_names() == ['RPM', 'MAP', 'EGT1']

    for name in ('RPM', 'MAP', 'EGT1'):
        widget._on_column_dropped_on_graph(0, name)
    _settle(qapp)
    plot_manager = widget.get_active_graph_container().plot_manager
    assert set(plot_manager.current_signals) == {'RPM_0', 'MAP_0', 'EGT1_0'}

    correlations = widget.correlations_panel_manager
    correlations._on_target_changed('RPM')
    correlations._calculate_correlations(force=True)
    assert set(correlations.current_correlations) == {'MAP', 'EGT1'}
    assert all(-1.0 <= r <= 1.0 for r in correlations.current_correlations.values())

    stats = processor.calculate_statistics(['RPM'], time_range=(5.0, 10.0))['RPM']
    assert 1500 < stats['min'] < stats['mean'] < stats['max'] < 2501

    widget._on_global_normalization_toggled(True)
    _settle(qapp, 0.2)
    widget._on_global_normalization_toggled(False)
    _settle(qapp, 0.2)

    widget._add_tab('Second')
    _settle(qapp, 0.2)
    config = widget.get_layout_config()
    assert config
    widget.set_layout_config(config)
    _settle(qapp)
