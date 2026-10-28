"""
Shared pytest setup.

Runs Qt without a display, puts the project root on sys.path and runs each
test session in a temporary folder (the older tests write their MPAI files
to the current directory). Modal message boxes are answered automatically
so a test never waits for a click.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@pytest.fixture(scope='session')
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope='session', autouse=True)
def _session_workdir(tmp_path_factory):
    # The loader caches converted MPAI files under %LOCALAPPDATA%; keep the
    # tests away from the user's real cache
    previous_cwd = os.getcwd()
    previous_appdata = os.environ.get('LOCALAPPDATA')
    os.environ['LOCALAPPDATA'] = str(tmp_path_factory.mktemp('appdata'))
    os.chdir(tmp_path_factory.mktemp('workdir'))
    yield
    os.chdir(previous_cwd)
    if previous_appdata is None:
        os.environ.pop('LOCALAPPDATA', None)
    else:
        os.environ['LOCALAPPDATA'] = previous_appdata


def write_engine_csv(path: Path) -> Path:
    """Write a small engine log: 20 s at 100 Hz with RPM, MAP and EGT1."""
    import numpy as np
    t = np.arange(0, 2000) * 0.01
    rpm = 2000 + 500 * np.sin(t)
    map_ = 80 + 5 * np.cos(t)
    egt = 600 + np.arange(t.size) * 0.01
    with open(path, 'w') as f:
        f.write('time,RPM,MAP,EGT1\n')
        for row in zip(t, rpm, map_, egt):
            f.write('%.2f,%.3f,%.3f,%.3f\n' % row)
    return path


def csv_import_settings(csv_path: Path) -> dict:
    """Import settings as the import dialog returns them for the engine CSV."""
    return {
        'file_path': str(csv_path), 'file_format': 'csv', 'encoding': 'utf-8',
        'delimiter': ',', 'decimal_comma': False, 'header_row': 0, 'start_row': 1,
        'time_mode': 'Mevcut Kolonu Kullan', 'create_custom_time': False,
        'time_column': 'time', 'time_column_original': 'time',
        'time_format': 'Otomatik', 'time_unit': 'saniye',
    }


@pytest.fixture(scope='session')
def engine_csv(tmp_path_factory):
    return write_engine_csv(tmp_path_factory.mktemp('data') / 'engine.csv')


@pytest.fixture(scope='session')
def engine_reader(qapp, engine_csv):
    """The engine CSV converted to MPAI by DataLoader, as the app loads it."""
    from src.data.data_loader import DataLoader
    loader = DataLoader(csv_import_settings(engine_csv))
    result = {}
    loader.finished.connect(lambda reader, time_col: result.update(reader=reader, time_col=time_col))
    loader.error.connect(lambda msg: result.update(error=msg))
    loader.run()
    assert 'error' not in result, result.get('error')
    return result['reader'], result['time_col']


@pytest.fixture(autouse=True)
def _auto_answer_message_boxes(monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(lambda *a, **k: QMessageBox.Yes))
    for name in ('information', 'warning', 'critical'):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, 'exec_', lambda self: QMessageBox.Ok)
