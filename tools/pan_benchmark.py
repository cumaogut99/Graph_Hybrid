"""
Measure how smoothly a graph pans on this computer.

Opens the application with a data file (through the import dialog's own
format detection), plots some channels, drags the first graph with the left
mouse button and reports the frame rate - with OpenGL drawing and without.

Usage (from the project folder):
    python tools/pan_benchmark.py path/to/file.csv
    python tools/pan_benchmark.py path/to/file.csv --graphs 3 --signals 4

Each mode runs in its own process (OpenGL is chosen before plots are made).
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_one(args):
    sys.path.insert(0, ROOT)
    import importlib.util

    import numpy as np
    import pyqtgraph as pg
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication, QMessageBox

    # No questions (unsaved project on close, data quality notes): unattended
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    for name in ('information', 'warning', 'critical'):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))

    spec = importlib.util.spec_from_file_location('app_main', os.path.join(ROOT, 'app.py'))
    app_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_main)  # sets up the application like app.py does
    pg.setConfigOptions(useOpenGL=args.opengl == 'on')
    app = QApplication.instance() or QApplication(sys.argv)

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.005)

    window = app_main.TimeGraphApp()
    window.resize(args.width, args.height)
    window.show()
    pump(0.5)

    from src.data.data_import_dialog import DataImportDialog
    dialog = DataImportDialog(os.path.abspath(args.file))
    end = time.time() + 60
    while dialog.preview_df is None and time.time() < end:
        pump(0.05)
    settings = dialog.get_import_settings()
    dialog.close()

    window.data_ops.load_data_with_settings(settings)
    end = time.time() + 600
    while not window.file_manager.loaded_files and time.time() < end:
        pump(0.1)
    if not window.file_manager.loaded_files:
        sys.exit('The file did not load')
    pump(1)

    widget = window.widget_container_manager.get_active_widget()
    if args.graphs > 1:
        widget._on_graph_count_changed(args.graphs)
        pump(1)
    names = widget.signal_processor.get_signal_names()
    for g in range(args.graphs):
        for name in names[g * args.signals:(g + 1) * args.signals]:
            widget._on_column_dropped_on_graph(g, name)
    pump(2)

    plot_manager = widget.get_active_graph_container().plot_manager
    plot = plot_manager.plot_widgets[0]
    x_min, x_max = plot.getViewBox().viewRange()[0]
    span = (x_max - x_min) * args.zoom
    plot.setXRange(x_min + span, x_min + 2 * span, padding=0)
    pump(1)

    from pyqtgraph.widgets.GraphicsView import GraphicsView
    paints = [0]
    original_paint = GraphicsView.paintEvent

    def counting_paint(view, event):
        if view is plot:
            paints[0] += 1
        return original_paint(view, event)

    GraphicsView.paintEvent = counting_paint

    viewport = plot.viewport()
    start = viewport.rect().center()

    def send(kind, pos, button, buttons):
        app.sendEvent(viewport, QMouseEvent(kind, QPointF(pos), button, buttons, Qt.NoModifier))

    send(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton)
    step_times = []
    t_start = time.perf_counter()
    for i in range(1, args.moves + 1):
        t0 = time.perf_counter()
        send(QEvent.MouseMove, start + QPoint(-args.step * i, 0), Qt.NoButton, Qt.LeftButton)
        app.processEvents()
        step_times.append((time.perf_counter() - t0) * 1000)
    total = time.perf_counter() - t_start
    send(QEvent.MouseButtonRelease, start + QPoint(-args.step * args.moves, 0), Qt.LeftButton, Qt.NoButton)
    pump(0.5)

    frames = max(1, paints[0])
    steps = np.array(step_times)
    print(f"OpenGL {args.opengl:3}: {frames / total:5.1f} fps, {total * 1000 / frames:6.1f} ms per frame "
          f"({frames} frames for {args.moves} mouse moves; slowest move {steps.max():.0f} ms)", flush=True)
    window.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', help='CSV, Excel, TDMS or MPAI file')
    parser.add_argument('--graphs', type=int, default=3, help='graphs in the tab (default 3)')
    parser.add_argument('--signals', type=int, default=4, help='signals per graph (default 4)')
    parser.add_argument('--zoom', type=float, default=0.25,
                        help='visible part of the data while panning (default 0.25)')
    parser.add_argument('--moves', type=int, default=150, help='mouse moves in the drag (default 150)')
    parser.add_argument('--step', type=int, default=4, help='pixels per mouse move (default 4)')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1000)
    parser.add_argument('--opengl', choices=['on', 'off'], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.opengl:
        _run_one(args)
        return
    print(f"Panning {args.graphs} graphs x {args.signals} signals of {os.path.basename(args.file)}")
    for mode in ('on', 'off'):
        result = subprocess.run([sys.executable, os.path.abspath(__file__), *sys.argv[1:], '--opengl', mode],
                                capture_output=True, text=True)
        lines = [line for line in result.stdout.splitlines() if line.startswith('OpenGL')]
        print(lines[-1] if lines else f"OpenGL {mode}: failed\n{result.stderr[-2000:]}")


if __name__ == '__main__':
    main()
