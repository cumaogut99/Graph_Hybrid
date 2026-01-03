# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller Spec Dosyası - Time Graph Uygulaması
================================================

Bu dosya Time Graph uygulamasını EXE formatına çevirmek için kullanılır.
Kullanım: pyinstaller time_graph_app.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Uygulama bilgileri
app_name = 'TimeGraphApp'
script_name = 'app.py'
icon_file = 'ikon.ico'

# Veri dosyalarını topla
datas = []

# İkon dosyalarını ekle
if os.path.exists('icons'):
    datas.append(('icons', 'icons'))

# lib/ klasöründeki Arrow/Parquet DLL'lerini ekle
if os.path.exists('lib'):
    datas.append(('lib', 'lib'))

# İkon dosyasını ekle
if os.path.exists(icon_file):
    datas.append((icon_file, '.'))

# Gerekli kütüphanelerin veri dosyalarını topla
try:
    # PyQt5 veri dosyaları
    datas += collect_data_files('PyQt5')
except:
    pass

try:
    # Matplotlib veri dosyaları
    datas += collect_data_files('matplotlib')
except:
    pass

try:
    # Add pandas and numpy data files
    datas += collect_data_files('polars')
    datas += collect_data_files('numpy')
    
    # Add matplotlib data files (e.g., fonts, styles)
except:
    pass

# Gizli importları topla
hiddenimports = []
try:
    # Collect all submodules for key libraries
    hiddenimports += collect_submodules('PyQt5')
    hiddenimports += collect_submodules('numpy')
    hiddenimports += collect_submodules('polars')
    hiddenimports += collect_submodules('matplotlib')
    hiddenimports += collect_submodules('pyqtgraph')
    hiddenimports += collect_submodules('scipy')
    
    # Add specific hidden imports that might be missed
    hiddenimports += [
        'scipy.special._cdflib',
        'scipy.integrate',
        'scipy.interpolate',
        'scipy.signal',
        'polars.libs'
    ]
    
    # Optional: Add any other necessary hidden imports
except:
    pass

# Ek gizli importlar
hiddenimports += [
    'PyQt5.QtCore',
    'PyQt5.QtGui', 
    'PyQt5.QtWidgets',
    'matplotlib.backends.backend_qt5agg',
    'pandas._libs.tslibs.base',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.timestamps',
    'vaex.core',
    'vaex.dataframe',
    'numpy.core._methods',
    'numpy.lib.format',
]

# Analiz aşaması
a = Analysis(
    [script_name],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'unittest', 
        'test',
        'distutils',
        'setuptools',
        'pip'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ arşivi
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# EXE dosyası
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI uygulaması için False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,  # İkon dosyası
    version_file=None,
)
