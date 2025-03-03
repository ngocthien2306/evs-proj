STYLE_SHEET = """
QMainWindow {
    background-color: #f0f0f0;
}

QGroupBox {
    background-color: white;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    margin-top: 1em;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    color: #2c3e50;
}

QPushButton {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
    min-width: 100px;
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:disabled {
    background-color: #bdc3c7;
}

QLineEdit {
    padding: 6px;
    border: 2px solid #e0e0e0;
    border-radius: 4px;
    background-color: white;
}

QSpinBox, QDoubleSpinBox {
    padding: 5px;
    border: 2px solid #e0e0e0;
    border-radius: 4px;
    background-color: white;
}

QLabel {
    color: #2c3e50;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
}

#status_label {
    font-weight: bold;
    color: #27ae60;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 4px;
}

QRadioButton {
    spacing: 8px;
    color: #2c3e50;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
}

QRadioButton::indicator:checked {
    background-color: #3498db;
    border: 2px solid #2980b9;
    border-radius: 9px;
}

QRadioButton::indicator:unchecked {
    background-color: white;
    border: 2px solid #bdc3c7;
    border-radius: 9px;
}
"""
