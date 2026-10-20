"""
Polars runtime selection for older CPUs
=======================================

The standard ``polars`` wheel is compiled for modern CPUs (AVX2, FMA, BMI...).
Many Pentium / Celeron processors lack these instructions and the process
crashes with an illegal instruction as soon as the Polars binary is loaded.

``polars-lts-cpu`` ships the same Python code with a binary compiled for older
CPUs. Only the native module (``polars/polars.pyd``) differs, so the build
bundles that one file from ``polars-lts-cpu`` (see tools/fetch_polars_compat.py)
and ``select_polars_runtime()`` loads it instead of the standard binary when
the CPU lacks a required feature. Modern CPUs keep the fast binary.

Must be called before anything imports polars. Setting the environment
variable TIMEGRAPH_POLARS_COMPAT=1 forces the compatible binary (to test it on
a modern machine).
"""

import importlib
import importlib.util
import os
import sys
from typing import List, Optional

COMPAT_DIR_NAME = 'polars_compat'

# 'fast', 'compat' or 'default' (no selection made / compat binary missing)
selected_runtime = 'default'


def _compat_dir() -> Optional[str]:
    """Folder holding the polars-lts-cpu native module, if bundled."""
    if getattr(sys, 'frozen', False):
        candidates = [os.path.join(getattr(sys, '_MEIPASS', ''), COMPAT_DIR_NAME)]
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [os.path.join(root, 'vendor', COMPAT_DIR_NAME)]
    for path in candidates:
        if os.path.isdir(path) and any(
            name.startswith('polars.') and name.endswith(('.pyd', '.so'))
            for name in os.listdir(path)
        ):
            return path
    return None


def _missing_cpu_features() -> List[str]:
    """CPU features the installed Polars binary needs but the CPU lacks.

    Uses Polars' own pure-Python CPUID check, which is safe to run before the
    binary is loaded.
    """
    cpu_check = importlib.import_module('polars._cpu_check')
    if getattr(cpu_check, '_POLARS_LTS_CPU', False):
        return []
    required = [f.lstrip('+') for f in cpu_check._POLARS_FEATURE_FLAGS.split(',') if f]
    supported = cpu_check._read_cpu_flags()
    return [f for f in required if not supported.get(f, False)]


def select_polars_runtime() -> str:
    """Import polars with the binary that fits this CPU and return the choice."""
    global selected_runtime
    if 'polars' in sys.modules:
        return selected_runtime

    compat_dir = _compat_dir()
    if compat_dir is None:
        return selected_runtime

    spec = importlib.util.find_spec('polars')
    if spec is None or spec.loader is None or not spec.submodule_search_locations:
        return selected_runtime

    # Create the package module without running polars/__init__.py, so that
    # polars._cpu_check can be imported before the native module is loaded.
    module = importlib.util.module_from_spec(spec)
    sys.modules['polars'] = module
    try:
        try:
            if os.environ.get('TIMEGRAPH_POLARS_COMPAT') == '1':
                missing = ['forced']
            else:
                missing = _missing_cpu_features()
        except Exception:
            # Could not read the CPU flags: the compatible binary works on
            # every CPU, a wrong guess the other way would crash.
            missing = ['unknown']

        if missing:
            # polars/__init__.py imports the native module as polars.polars,
            # which is searched in __path__ order: put the compatible binary
            # first. Its own CPU check would warn about the standard flags.
            module.__path__.insert(0, compat_dir)
            os.environ['POLARS_SKIP_CPU_CHECK'] = '1'
            selected_runtime = 'compat'
        else:
            selected_runtime = 'fast'

        spec.loader.exec_module(module)
    except BaseException:
        for name in [n for n in sys.modules if n == 'polars' or n.startswith('polars.')]:
            del sys.modules[name]
        selected_runtime = 'default'
        raise
    return selected_runtime
