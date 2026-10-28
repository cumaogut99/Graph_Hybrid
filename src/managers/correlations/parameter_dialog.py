"""
Dialog to pick the target parameter, with a search box.

Moved from src/managers/correlations_panel_manager.py without changes.
"""

from typing import List, Optional
from PyQt5.QtWidgets import QVBoxLayout, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox, QLineEdit


class ParameterSelectionDialog(QDialog):
    """A dialog for searching and selecting a parameter."""
    def __init__(self, parameters: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Target Parameter")
        self.setMinimumWidth(300)
        # Apply theme styling from parent
        if hasattr(parent, 'theme_manager'):
            theme_colors = parent.theme_manager.get_theme_colors()
            is_light_theme = theme_colors.get('text_primary', '#ffffff') == '#212121'
            
            if is_light_theme:
                bg_color = theme_colors.get('background', '#ffffff')
                text_color = theme_colors.get('text_primary', '#212121')
                surface_color = theme_colors.get('surface', '#f5f5f5')
                border_color = theme_colors.get('border', '#cccccc')
                primary_color = theme_colors.get('primary', '#2196f3')
            else:
                bg_color = theme_colors.get('background', '#1a2332')
                text_color = theme_colors.get('text_primary', '#e6f3ff')
                surface_color = theme_colors.get('surface', '#2d4a66')
                border_color = theme_colors.get('border', '#4a90e2')
                primary_color = theme_colors.get('primary', '#4a90e2')
            
            self.setStyleSheet(f"""
                QDialog {{
                    background-color: {bg_color};
                    color: {text_color};
                }}
                QLineEdit {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                    border-radius: 4px;
                    padding: 6px;
                    color: {text_color};
                }}
                QListWidget {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                    border-radius: 4px;
                    color: {text_color};
                }}
                QListWidget::item:selected {{
                    background-color: {primary_color};
                    color: white;
                }}
                QPushButton {{
                    background-color: {primary_color};
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                }}
            """)
        else:
            self.setStyleSheet(parent.styleSheet())

        self.all_parameters = parameters
        self.selected_parameter = None

        layout = QVBoxLayout(self)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_box)

        self.list_widget = QListWidget()
        self.list_widget.addItems(self.all_parameters)
        self.list_widget.itemDoubleClicked.connect(self._on_item_selected)
        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _filter_list(self, text: str):
        self.list_widget.clear()
        if not text:
            self.list_widget.addItems(self.all_parameters)
        else:
            filtered_items = [p for p in self.all_parameters if text.lower() in p.lower()]
            self.list_widget.addItems(filtered_items)

    def _on_item_selected(self, item: QListWidgetItem):
        self.accept()

    def accept(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            self.selected_parameter = selected_item.text()
        super().accept()

    @staticmethod
    def get_parameter(parameters: List[str], parent=None) -> Optional[str]:
        dialog = ParameterSelectionDialog(parameters, parent)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_parameter
        return None
