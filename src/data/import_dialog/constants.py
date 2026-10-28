"""
Styles and choices of the import dialog.

Moved from src/data/data_import_dialog.py without changes.
"""

from src.utils.resources import icon_url

# [-] value [+] spin boxes, like the graph count in the toolbar
STEPPER_STYLE = """
            QSpinBox#stepper {
                padding: 2px 2px;  /* Qt reserves the -/+ button widths itself */
                min-width: 110px;
            }
            QSpinBox#stepper::up-button, QSpinBox#stepper::down-button {
                subcontrol-origin: border;
                background: transparent;
                border: none;
                border-radius: 4px;
                width: 26px;
                height: 22px;  /* full spinbox height inside the border */
            }
            QSpinBox#stepper::up-button {
                subcontrol-position: right;
            }
            QSpinBox#stepper::down-button {
                subcontrol-position: left;
            }
            QSpinBox#stepper::up-button:hover, QSpinBox#stepper::down-button:hover {
                background: rgba(127, 179, 211, 0.25);
            }
            QSpinBox#stepper::up-button:pressed, QSpinBox#stepper::down-button:pressed {
                background: rgba(127, 179, 211, 0.4);
            }
            QSpinBox#stepper::up-arrow {
                image: url(%s);
                width: 13px;
                height: 13px;
            }
            QSpinBox#stepper::down-arrow {
                image: url(%s);
                width: 13px;
                height: 13px;
            }
""" % (icon_url('spin-plus'), icon_url('spin-minus'))

# "Zaman Formatı" choices -> setting passed to the converter
TIME_FORMAT_CHOICES = {
    'Otomatik - Kendisi seçsin': 'Otomatik',
    '%Y-%m-%d %H:%M:%S (2024-01-15 14:30:45)': '%Y-%m-%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S (15/01/2024 14:30:45)': '%d/%m/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M:%S (01/15/2024 14:30:45)': '%m/%d/%Y %H:%M:%S',
    '%m/%d/%Y %I:%M:%S %p (01/15/2024 02:30:45 PM)': '%m/%d/%Y %I:%M:%S %p',
    '%d.%m.%Y %H:%M:%S (15.01.2024 14:30:45)': '%d.%m.%Y %H:%M:%S',
    '%Y-%m-%d (2024-01-15)': '%Y-%m-%d',
    '%d/%m/%Y (15/01/2024)': '%d/%m/%Y',
    'Unix Timestamp (1704110400)': 'Unix Timestamp',
    'Saniyelik Index (0.0, 0.1, 0.2...)': 'Saniyelik Index',
}
