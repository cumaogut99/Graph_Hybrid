"""
Context menu, copying plots to the clipboard and time markers.

Part of PlotManager; moved from src/managers/plot_manager.py without changes.
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QAction
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from src.managers.plot import DateTimeAxisItem
if TYPE_CHECKING:
    from src.ui.time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)


class PlotMarkersMixin:
    """Context menu, copying plots to the clipboard and time markers."""

    def _setup_context_menu_for_plot(self, plot_widget, plot_index: int):
        """Setup the plot context menu, including a copy action for the whole tab."""
        view_box = plot_widget.getViewBox()
        menu = view_box.menu
        
        # Add separator before custom actions
        menu.addSeparator()
        
        # Add "Zoom to Cursors" action
        zoom_to_cursors_action = QAction("🎯 Zoom to Cursors", menu)
        zoom_to_cursors_action.setToolTip("Zoom to the range between dual cursors")
        
        # Get cursor_manager - PlotManager.parent is GraphContainer
        def get_cursor_manager():
            cursor_manager = None
            main_widget = None
            
            if hasattr(self.parent, 'main_widget'):
                main_widget = self.parent.main_widget
            
            if main_widget and hasattr(main_widget, 'cursor_manager'):
                cursor_manager = main_widget.cursor_manager
            
            return cursor_manager
        
        # Connect action
        def on_zoom_to_cursors():
            cursor_manager = get_cursor_manager()
            if cursor_manager and hasattr(cursor_manager, 'zoom_to_cursors'):
                success = cursor_manager.zoom_to_cursors()
                if success:
                    logger.info("Zoomed to cursors from context menu")
                else:
                    logger.warning("Failed to zoom to cursors")
        
        zoom_to_cursors_action.triggered.connect(on_zoom_to_cursors)
        
        # Update button state when menu is about to show
        def update_menu():
            cursor_manager = get_cursor_manager()
            if cursor_manager and hasattr(cursor_manager, 'can_zoom_to_cursors'):
                can_zoom = cursor_manager.can_zoom_to_cursors()
                zoom_to_cursors_action.setEnabled(can_zoom)
            else:
                zoom_to_cursors_action.setEnabled(False)
        
        menu.aboutToShow.connect(update_menu)
        menu.addAction(zoom_to_cursors_action)

        copy_action = QAction("Copy", menu)
        copy_action.setToolTip("Copy all charts in this tab to the clipboard")
        copy_action.triggered.connect(self.copy_all_plots_to_clipboard)
        menu.addAction(copy_action)

        # "Add Marker" at the x position where the menu was opened
        original_raise_menu = view_box.raiseContextMenu

        def raise_menu(ev):
            self._context_menu_x = view_box.mapSceneToView(ev.scenePos()).x()
            original_raise_menu(ev)

        view_box.raiseContextMenu = raise_menu

        add_marker_action = QAction("📍 Add Marker", menu)
        add_marker_action.triggered.connect(
            lambda: self._context_menu_x is not None and self.add_marker(self._context_menu_x))
        menu.addAction(add_marker_action)
        
        logger.debug(f"Custom context menu setup for plot {plot_index}")

    def copy_all_plots_to_clipboard(self) -> bool:
        """Copy all charts, including their scene overlays, as one clipboard image."""
        pixmap = self._render_all_plots_to_pixmap()
        if pixmap is None or pixmap.isNull():
            logger.warning("Cannot copy charts: failed to render plot scenes")
            return False

        QApplication.clipboard().setPixmap(pixmap)
        logger.info("Copied all charts in the current tab to the clipboard")
        return True

    def _render_all_plots_to_pixmap(self) -> Optional[QPixmap]:
        """Render plot scenes directly so OpenGL viewports do not omit plot items."""
        if self.plot_container is None or not self.plot_widgets:
            logger.warning("Cannot render charts: plot container is unavailable")
            return None

        size = self.plot_container.size()
        if size.width() <= 0 or size.height() <= 0:
            logger.warning("Cannot render charts: plot container has no size")
            return None

        image = QImage(size, QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(self.theme_colors['background']))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)

        try:
            for plot_widget in self.plot_widgets:
                if not plot_widget.isVisible():
                    continue

                # Rendering the scene (rather than grabbing the OpenGL viewport)
                # includes PlotDataItems, InfiniteLines, legends, and other overlays.
                target = QRectF(plot_widget.geometry())
                source = plot_widget.mapToScene(plot_widget.viewport().rect()).boundingRect()
                plot_widget.scene().render(painter, target, source, Qt.IgnoreAspectRatio)
        finally:
            painter.end()

        return QPixmap.fromImage(image)

    def add_marker(self, x: float) -> int:
        """Add a numbered vertical marker at x on every plot of this tab."""
        number = self._next_marker_number
        self._next_marker_number += 1
        self.markers.append({'number': number, 'x': float(x)})
        self._draw_markers()
        self.markers_changed.emit()
        logger.info(f"Marker {number} added at x={x}")
        return number

    def remove_marker(self, number: int):
        """Remove the marker with this number (numbers of others stay the same)."""
        self.markers = [m for m in self.markers if m['number'] != number]
        self._draw_markers()
        self.markers_changed.emit()

    def clear_markers(self):
        """Remove all markers and start numbering from 1 again."""
        if not self.markers:
            return
        self.markers = []
        self._next_marker_number = 1
        self._draw_markers()
        self.markers_changed.emit()

    def get_markers(self) -> List[Dict[str, Any]]:
        return list(self.markers)

    def format_x_value(self, x: float) -> str:
        """x as shown on the time axis (clock/date when the datetime axis is on)."""
        if self.plot_widgets:
            axis = self.plot_widgets[0].getAxis('bottom')
            if isinstance(axis, DateTimeAxisItem) and axis.is_datetime_axis:
                return axis.tickStrings([x], 1.0, 0.001)[0]
        return f"{x:.6g}"

    def _draw_markers(self):
        """(Re)create the marker lines on all current plot widgets."""
        for plot_widget, line in self._marker_items:
            try:
                plot_widget.removeItem(line)
            except Exception:
                pass  # already removed by plot_widget.clear() or a rebuild
        self._marker_items = []

        pen = pg.mkPen(color=self.MARKER_COLOR, width=2, style=Qt.DashLine)
        for marker in self.markers:
            for plot_widget in self.plot_widgets:
                line = pg.InfiniteLine(
                    pos=marker['x'], angle=90, movable=False, pen=pen,
                    label=str(marker['number']),
                    labelOpts={'position': 0.95, 'color': self.MARKER_COLOR,
                               'fill': (0, 0, 0, 160), 'movable': False},
                )
                line.setZValue(50)
                plot_widget.addItem(line, ignoreBounds=True)
                self._marker_items.append((plot_widget, line))
