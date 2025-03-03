
from base64 import b64decode, b64encode
import hashlib
import json
import os
import time
from PyQt5.QtWidgets import (QVBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QMessageBox, QDialog)
from PyQt5.QtCore import Qt

class CredentialsManager:
    def __init__(self, base_path):
        self.app_dir = base_path
        self.credentials_file = os.path.join(self.app_dir, "credentials.json")
        self.initialize_credentials()
    
    def initialize_credentials(self):
        """Initialize credentials if file doesn't exist"""
        try:
            if not os.path.exists(self.credentials_file):
                # Try to create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
                
                salt = os.urandom(32)
                password = "dlmsllab321@"
                hashed = self.hash_password(password, salt)
                
                credentials = {
                    "deeplab": {
                        "salt": b64encode(salt).decode('utf-8'),
                        "hash": b64encode(hashed).decode('utf-8')
                    }
                }
                
                try:
                    with open(self.credentials_file, 'w') as f:
                        json.dump(credentials, f, indent=4)
                    print(f"Created credentials file at: {self.credentials_file}")
                except PermissionError:
                    print("Warning: Could not create credentials file - Permission denied")
                except Exception as e:
                    print(f"Warning: Could not create credentials file - {str(e)}")
        except Exception as e:
            print(f"Error initializing credentials: {e}")
    
    def hash_password(self, password: str, salt: bytes) -> bytes:
        """Hash password with salt using PBKDF2"""
        return hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt, 
            100000  # Number of iterations
        )
    
    def verify_credentials(self, username: str, password: str) -> bool:
        """Verify username and password"""
        try:
            print(f"Verifying credentials for username: {username}")
            
            with open(self.credentials_file, 'r') as f:
                credentials = json.load(f)
            print(f"Loaded credentials: {credentials.keys()}")
            
            if username not in credentials:
                print(f"Username {username} not found")
                return False
            
            stored = credentials[username]
            salt = b64decode(stored['salt'])
            stored_hash = b64decode(stored['hash'])
            
            hashed = self.hash_password(password, salt)
            
            result = stored_hash == hashed
            print(f"Verification result: {result}")
            return result
                
        except Exception as e:
            print(f"Error verifying credentials: {e}")
            return False
    
class LoginDialog(QDialog):
    MAX_ATTEMPTS = 3
    LOCKOUT_TIME = 300  # 5 minutes
    
    def __init__(self, base_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Admin Login")
        self.setModal(True)
        self.setFixedWidth(300)
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                margin: 5px 0px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 100px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        self.credentials_manager = CredentialsManager(base_path)
        self.attempt_count = 0
        self.last_attempt_time = 0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Admin Authentication")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title, alignment=Qt.AlignCenter)
        
        # Username
        self.username = QLineEdit()
        self.username.setPlaceholderText("Username")
        layout.addWidget(self.username)
        
        # Password
        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.Password)
        self.password.returnPressed.connect(self.check_credentials)  # Allow Enter key
        layout.addWidget(self.password)
        
        # Login button
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.check_credentials)
        layout.addWidget(self.login_btn)
        
        self.setLayout(layout)
    
    def check_lockout(self) -> bool:
        if self.attempt_count >= self.MAX_ATTEMPTS:
            time_since_last = time.time() - self.last_attempt_time
            if time_since_last < self.LOCKOUT_TIME:
                remaining = int(self.LOCKOUT_TIME - time_since_last)
                QMessageBox.warning(
                    self,
                    "Account Locked",
                    f"Too many failed attempts. Please try again in {remaining} seconds."
                )
                return True
            else:
                self.attempt_count = 0
        return False
    
    def check_credentials(self):
        if self.check_lockout():
            return
            
        username = self.username.text().strip()
        password = self.password.text()
        
        if not username or not password:
            QMessageBox.warning(self, "Error", "Please enter both username and password")
            return
        
        if self.credentials_manager.verify_credentials(username, password):
            self.accept()
        else:
            self.attempt_count += 1
            self.last_attempt_time = time.time()
            remaining_attempts = self.MAX_ATTEMPTS - self.attempt_count
            
            QMessageBox.warning(
                self, 
                "Error", 
                f"Invalid credentials. {remaining_attempts} attempts remaining."
            )
            
            self.password.clear()   
