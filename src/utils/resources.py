"""
Paths to bundled files (icons etc.) that work both from source and from the
PyInstaller exe. A onefile exe unpacks its data into sys._MEIPASS, not into
the working directory, so relative paths like 'icons/x.svg' break there.
"""

import os
import sys

_BASE_DIR = getattr(
    sys, '_MEIPASS',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)


def resource_path(*parts: str) -> str:
    """Absolute path of a file shipped with the app (relative to the project root)."""
    return os.path.join(_BASE_DIR, *parts)


def icon_url(name: str) -> str:
    """Path of icons/<name>.svg for use inside a Qt stylesheet url(...)."""
    return resource_path('icons', f'{name}.svg').replace(os.sep, '/')


def resolve_icon_urls(stylesheet: str) -> str:
    """Rewrite every url(icons/...) in a stylesheet to the bundled icon path."""
    return stylesheet.replace('url(icons/', 'url(' + resource_path('icons').replace(os.sep, '/') + '/')
