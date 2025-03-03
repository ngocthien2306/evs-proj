from PyQt5.QtWidgets import (QVBoxLayout,  QLabel, QDialog, QProgressBar)
from PyQt5.QtCore import Qt, QTimer

class LoadingDialog(QDialog):
    def __init__(self, parent=None, message="Processing...", duration=3000):
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setModal(True)  
        self.setFixedSize(300, 150)
        self.setWindowFlags(
            Qt.Window | 
            Qt.WindowStaysOnTopHint | 
            Qt.CustomizeWindowHint
        )
        
        layout = QVBoxLayout()
        
        # Message label
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress bar
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
        # Style
        self.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 10px;
            }
            QLabel {
                font-size: 14px;
                color: #2c3e50;
            }
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        
        # Timer to auto-close dialog
        self.close_timer = QTimer(self)
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.close)
        self.close_timer.start(duration)
