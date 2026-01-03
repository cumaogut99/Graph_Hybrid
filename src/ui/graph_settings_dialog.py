from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QListWidget, QDialogButtonBox,
                             QListWidgetItem, QLabel, QFrame, QLineEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class GraphSettingsDialog(QDialog):
    """
    A dialog to select which signals to display on a specific graph plot.
    """
    def __init__(self, graph_index, all_signals, visible_signals, parent=None):
        """
        Initializes the dialog.

        Args:
            graph_index (int): The index of the graph being configured.
            all_signals (list): A list of all available signal names.
            visible_signals (list): A list of signal names currently visible on the graph.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Graph {graph_index + 1} Settings")
        self.setMinimumWidth(350)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a2332;
                color: #e6f3ff;
            }
            QLabel {
                font-size: 14px;
                padding-bottom: 5px;
                color: #ffffff;
            }
            QListWidget {
                background-color: #2d4a66;
                border: 1px solid #4a90e2;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                color: #e6f3ff;
            }
            QListWidget::item:hover {
                background-color: #3a5f7a;
            }
            QListWidget::item:selected {
                background-color: #4a90e2;
                color: white;
            }
        """)

        self.all_signals = sorted(all_signals)
        self.visible_signals = set(visible_signals)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title_label = QLabel(f"Select signals to display on Graph {graph_index + 1}:")
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search signals...")
        self.search_bar.textChanged.connect(self._filter_signals)
        layout.addWidget(self.search_bar)
        
        # Signal list with checkboxes
        self.signal_list = QListWidget()
        for signal_name in self.all_signals:
            item = QListWidgetItem(signal_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if signal_name in self.visible_signals:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.signal_list.addItem(item)
        layout.addWidget(self.signal_list)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _filter_signals(self, text):
        """
        Filters the signal list based on the search text.
        """
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            # Hide item if it does not match the search text (case-insensitive)
            item.setHidden(text.lower() not in item.text().lower())

    def get_selected_signals(self):
        """
        Returns the list of signal names that were checked by the user.

        Returns:
            list: A list of selected signal names.
        """
        selected = []
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected
