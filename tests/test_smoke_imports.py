"""Every application module imports without errors."""

import importlib
import importlib.util

import pytest

from conftest import ROOT

OPTIONAL_DEPENDENCIES = {'pyarrow'}


def _project_modules():
    names = ['time_graph_widget']
    for base in ('src', 'app'):
        for path in sorted((ROOT / base).rglob('*.py')):
            if '__pycache__' in path.parts:
                continue
            parts = list(path.relative_to(ROOT).with_suffix('').parts)
            if parts[-1] == '__init__':
                parts.pop()
            names.append('.'.join(parts))
    return names


@pytest.mark.parametrize('name', _project_modules())
def test_module_imports(qapp, name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name in OPTIONAL_DEPENDENCIES:
            pytest.skip(f'optional dependency {e.name} not installed')
        raise


def test_main_script_imports(qapp):
    spec = importlib.util.spec_from_file_location('app_main', ROOT / 'app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, 'TimeGraphApp')
