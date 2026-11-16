"""Cursors and pan/zoom: positions are emitted only when a cursor moves."""

import time

import pyqtgraph as pg

from src.managers.cursor_manager import CursorManager


def _manager(qapp):
    plot = pg.PlotWidget()
    plot.show()  # cursors are created on a visible plot
    qapp.processEvents()
    cm = CursorManager([plot])
    cm.set_mode('dual')
    end = time.time() + 2  # cursors are created by a 50 ms timer
    while not (cm.dual_cursors_1 and cm.dual_cursors_2) and time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)
    emitted = []
    cm.cursor_moved.connect(emitted.append)
    assert cm.dual_cursors_1 and cm.dual_cursors_2
    return cm, emitted


def test_pan_with_cursors_in_view_emits_nothing(qapp):
    cm, emitted = _manager(qapp)
    c1, c2 = cm.dual_cursors_1[0].value(), cm.dual_cursors_2[0].value()
    low, high = min(c1, c2), max(c1, c2)
    cm._on_view_range_changed(None, [[low - 10, high + 10], [0, 1]])
    assert emitted == []


def test_cursor_pushed_to_the_view_edge_is_emitted(qapp):
    cm, emitted = _manager(qapp)
    c1, c2 = cm.dual_cursors_1[0].value(), cm.dual_cursors_2[0].value()
    high = max(c1, c2)
    cm._on_view_range_changed(None, [[high + 5, high + 20], [0, 1]])
    assert len(emitted) == 1
    assert all(v == high + 5 for v in emitted[0].values())
