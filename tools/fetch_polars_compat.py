"""
Fetch the polars-lts-cpu native module for older CPUs
=====================================================

Downloads the ``polars-lts-cpu`` wheel matching the installed ``polars``
version and extracts its native module into ``vendor/polars_compat/``.
time_graph_app.spec bundles that folder and src/utils/polars_runtime.py loads
it on CPUs without AVX2/FMA (Pentium, Celeron...).

Run once before building the EXE, and again after upgrading polars:
    python tools/fetch_polars_compat.py
"""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from importlib.metadata import version
except ImportError:  # Python < 3.8
    from importlib_metadata import version

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / 'vendor' / 'polars_compat'


def main() -> int:
    polars_version = version('polars')
    print(f"Installed polars: {polars_version}")

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'download',
            f'polars-lts-cpu=={polars_version}',
            '--no-deps', '--only-binary=:all:', '-d', tmp,
        ])
        wheels = list(Path(tmp).glob('polars_lts_cpu-*.whl'))
        if not wheels:
            print("polars-lts-cpu wheel not found", file=sys.stderr)
            return 1

        with zipfile.ZipFile(wheels[0]) as wheel:
            native = [
                name for name in wheel.namelist()
                if name.startswith('polars/polars.') and name.endswith(('.pyd', '.so'))
            ]
            if len(native) != 1:
                print(f"Unexpected wheel layout: {native}", file=sys.stderr)
                return 1

            if TARGET.exists():
                shutil.rmtree(TARGET)
            TARGET.mkdir(parents=True)
            out_path = TARGET / Path(native[0]).name
            with wheel.open(native[0]) as src, open(out_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)

    (TARGET / 'VERSION').write_text(polars_version, encoding='utf-8')
    print(f"Wrote {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
