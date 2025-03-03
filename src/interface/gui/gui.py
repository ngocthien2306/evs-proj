import json
from queue import Empty
import time
import traceback
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
                           QGroupBox, QFormLayout, QMessageBox, QScrollArea, 
                           QSizePolicy, QComboBox, QRadioButton, QDialog
                           )
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEventLoop
from PyQt5.QtGui import QImage, QPixmap

from src.infrastructure.utils.system_utils import restart_usb_devices
from src.interface.gui.dialog.loading_dialog import LoadingDialog
from src.interface.gui.dialog.login_dialog import LoginDialog
from src.interface.gui.styles import STYLE_SHEET
from src.application.services.main_service import LogicFactoryService
from src.domain.value_objects.config import RadarConfig
from src.domain.entities.entities import ProcessingParams
from src.infrastructure.utils.logging_config import setup_logger

import os
os.environ['CUPY_ACCELERATORS'] = 'cutensor'
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1'

logger = setup_logger('gui_logger')

    



class ProcessingThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, base_path, config, input_path, output_path, params: ProcessingParams):
        super().__init__()
        self.config = config
        self.base_path = base_path
        self.input_path = input_path
        self.output_path = output_path  
        self.params = params
        self.running = True
        
        self.video_writer = None
        self.combined_width = 1920
        self.combined_height = 1080

    def run(self):
        self.stream_processor = None
        video_writer = None
        
        try:
            combined_width = None
            combined_height = None

            if self.output_path and self.output_path.strip():
                try:
                    output_dir = os.path.dirname(self.output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)

                except Exception as e:
                    print(f"Output directory error: {str(e)}")
                    self.finished.emit(False, f"Output directory error: {str(e)}")
                    return

            # Rest of the code remains the same
            self.stream_processor = LogicFactoryService(
                self.base_path,
                frame_queue_size=5,
                resize_dims=(self.params.camera_width, self.params.camera_height),
                target_fps=30,
                input_filename=self.input_path,
                args=self.params
            )

            if self.input_path != "":
                print("Run file")
                result_queue = self.stream_processor.start_ui_one_thread()
            else:
                print("Run camera")
                result_queue = self.stream_processor.start_ui()

            last_result = None
            frame_count = 0
            frames_to_write = []  
            
            while self.running:
                try:
                    try:
                        result = result_queue.get(timeout=0.1)
                        
                        if isinstance(result[0], str) and result[0] == "EOF":
                            print("Reached end of file")
                            break
                        
                        last_result = result
                    except Empty:
                        if last_result is None:
                            time.sleep(0.05)
                            continue
                        result = last_result

                    tracked_frame, _, _, _, _ = result

                    if combined_width is None or combined_height is None:
                        combined_height, combined_width = tracked_frame.shape[:2]
                        print(f"Detected frame size: {combined_width}x{combined_height}")

                        if self.output_path and self.output_path.strip():
                            try:
                                codecs = [
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    cv2.VideoWriter_fourcc(*'avc1'),
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    cv2.VideoWriter_fourcc(*'MJPG')
                                ]

                                video_writer = None
                                for codec in codecs:
                                    try:
                                        video_writer = cv2.VideoWriter(
                                            self.output_path,
                                            codec,
                                            20.0,  # FPS
                                            (combined_width, combined_height),
                                            True  
                                        )
                                        
                                        if video_writer.isOpened():
                                            print(f"Successfully opened video writer with codec {codec}")
                                            break
                                    except Exception as codec_error:
                                        print(f"Failed with codec {codec}: {codec_error}")

                                if video_writer is None or not video_writer.isOpened():
                                    raise IOError("Could not open video writer with any codec")
                            
                            except Exception as e:
                                print(f"Video writer error: {str(e)}")
                                self.finished.emit(False, f"Video writer error: {str(e)}")
                                return

                    # Emit frame for UI display
                    self.frame_ready.emit(tracked_frame)
                    
                    # Collect frames to write
                    if video_writer is not None:
                        try:
                            frames_to_write.append(tracked_frame)
                            frame_count += 1
                        except Exception as write_error:
                            print(f"Frame preparation error: {write_error}")

                    if not self.running:
                        break

                except Exception as processing_error:
                    print(f"Processing loop error: {processing_error}")
                    break

            # Write collected frames
            if video_writer is not None and frames_to_write:
                print(f"Writing {len(frames_to_write)} frames to video")
                for frame in frames_to_write:
                    video_writer.write(frame)

            # Successful completion
            self.finished.emit(True, f"Processing completed. Frames processed: {frame_count}")

        except Exception as e:
            error_message = f"Unexpected error during processing: {str(e)}"
            print(error_message)
            traceback.print_exc()
            self.finished.emit(False, error_message)

        finally:
            try:
                # Stop stream processor
                if self.stream_processor:
                    self.stream_processor.stop()
                
                # Release video writer
                if video_writer:
                    video_writer.release()
                
                # Verify file
                if self.output_path and os.path.exists(self.output_path):
                    file_size = os.path.getsize(self.output_path)
                    print(f"Video file saved. Size: {file_size} bytes")
                    
                    try:
                        cap = cv2.VideoCapture(self.output_path)
                        if cap.isOpened():
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            print(f"Total frames in video: {total_frames}")
                            cap.release()
                        else:
                            print("Cannot open the saved video file")
                    except Exception as verify_error:
                        print(f"Video verification error: {verify_error}")
                else:
                    print(f"Video file not found at: {self.output_path}")
            
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")     
                      
    def stop(self):
        self.running = False
        
class EventProcessorGUI(QMainWindow):
    def __init__(self, base_path=""):
        super().__init__()
        self.is_admin = False
        self.setWindowTitle("WQ Tivi - Deep Learning Laboratory")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet(STYLE_SHEET)
        self.base_path = base_path
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setSpacing(20)
        
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMaximumWidth(400)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        self.create_io_section(left_layout)
        self.create_processing_section(left_layout)
        self.create_camera_section(left_layout)
        self.create_model_section(left_layout)
        self.create_performance_section(left_layout)
        self.create_control_section(left_layout)
        
        left_scroll.setWidget(left_widget)
        layout.addWidget(left_scroll)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        self.admin_link = QLabel()  # Save reference to admin link
        self.admin_link.setText('<a href="#" style="color: #3498db;">Is Admin?</a>')
        self.admin_link.setOpenExternalLinks(False)
        self.admin_link.linkActivated.connect(self.handle_admin_click)
        self.admin_link.setVisible(False)
        right_layout.addWidget(self.admin_link, alignment=Qt.AlignRight)
        
        
        self.create_display_section(right_layout)
        layout.addWidget(right_panel)
        
        self.processing_thread = None
        self.config = RadarConfig()
        
        if not self.check_admin_file():
            self.update_ui_visibility()
            
        self.loading_dialog = None
    
    def check_admin_file(self):
        try:
            admin_file = os.path.join(self.base_path, "admin.json")
            if os.path.exists(admin_file):
                with open(admin_file, 'r') as f:
                    admin_data = json.load(f)
                    if admin_data.get('username') == 'deeplab' and admin_data.get('password') == 'dlmsllab321@':
                        self.is_admin = True
                        self.show_fps.setChecked(True)
                        self.show_timing.setChecked(True)
                        return True
            return False
        except Exception as e:
            print(f"Error checking admin file: {e}")
            return False
    
    def handle_admin_click(self):
        if self.is_admin:
            # Handle logout
            self.is_admin = False
            self.show_fps.setChecked(False)
            self.show_timing.setChecked(False)
            self.admin_link.setText('<a href="#" style="color: #3498db;">Is Admin?</a>')
            self.update_ui_visibility()
            QMessageBox.information(self, "Success", "Logged out successfully")
        else:
            # Handle login
            self.show_login()
    
    def show_login(self):
        dialog = LoginDialog(self.base_path, self)
        if dialog.exec_() == QDialog.Accepted:
            self.is_admin = True
            self.show_fps.setChecked(True)
            self.show_timing.setChecked(True)
            self.admin_link.setText('<a href="#" style="color: #3498db;">Logout</a>')
            self.update_ui_visibility()
            QMessageBox.information(self, "Success", "Logged in as admin")
            
    def update_ui_visibility(self):
        # Show/hide advanced settings based on admin status
        # Processing Parameters
        for i in range(self.proc_layout.rowCount()):
            label_item = self.proc_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.proc_layout.itemAt(i, QFormLayout.FieldRole)
            
            if label_item and field_item:
                label = label_item.widget()
                field = field_item.widget()
                
                # Always show Output Height
                if label.text() == "Output Height:":
                    continue
                    
                # Show/hide other parameters based on admin status
                label.setVisible(self.is_admin)
                field.setVisible(self.is_admin)
        
        # Model Settings
        self.model_group.setVisible(self.is_admin)
        
        # Performance Metrics
        self.perf_group.setVisible(self.is_admin)
        # pass

    def create_io_section(self, parent_layout):
        io_group = QGroupBox("Input/Output")
        io_layout = QVBoxLayout()
        
        # Input source selection
        input_source_layout = QHBoxLayout()
        self.camera_radio = QRadioButton("Camera")
        self.file_radio = QRadioButton("Input File")
        self.camera_radio.setChecked(True)
        
        input_source_layout.addWidget(self.camera_radio)
        input_source_layout.addWidget(self.file_radio)
        io_layout.addLayout(input_source_layout)
        
        # Input file selection
        input_file_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Input file...")
        self.input_path.setEnabled(False)
        
        browse_btn = QPushButton("Browse")
        browse_btn.setMaximumWidth(80)
        browse_btn.clicked.connect(self.browse_input_file)
        browse_btn.setEnabled(False)
        self.browse_input_btn = browse_btn
        
        input_file_layout.addWidget(self.input_path)
        input_file_layout.addWidget(browse_btn)
        io_layout.addLayout(input_file_layout)
        
        # Add spacing between input and output
        # io_layout.addSpacing(10)  # Add 10 pixels spacing
        
        # Output file selection
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Output file (output.mp4)")
        
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.setMaximumWidth(80)
        output_browse_btn.clicked.connect(self.browse_output_file)
        
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_browse_btn)
        io_layout.addLayout(output_layout)
        
        # Connect signals
        self.camera_radio.toggled.connect(self.on_input_source_changed)
        self.file_radio.toggled.connect(self.on_input_source_changed)
        
        io_group.setLayout(io_layout)
        parent_layout.addWidget(io_group)
        
    def on_input_source_changed(self):
        is_file_mode = self.file_radio.isChecked()
        self.input_path.setEnabled(is_file_mode)
        self.browse_input_btn.setEnabled(is_file_mode)
        
        if not is_file_mode:
            self.input_path.clear()  # Clear input path when switching to camera

    def create_processing_section(self, parent_layout):
        proc_group = QGroupBox("Processing Parameters")
        self.proc_layout = QFormLayout()  # Save reference to layout

        
        self.output_height = QSpinBox()
        self.output_height.setRange(480, 4096)
        self.output_height.setValue(720)
        self.output_height.setSingleStep(120)
        self.proc_layout.addRow("Output Height:", self.output_height)
        
        params = [
            ("Trim Angle:", "trim_angle", QDoubleSpinBox, (0.1, 10.0, 1.0)),
            ("Max Objects:", "max_objects", QSpinBox, (1, 10, 3)),
            ("Neighborhood Size:", "neighborhood_size", QSpinBox, (1, 10, 3)),
            ("Time Tolerance:", "time_tolerance", QSpinBox, (100, 5000, 1000)),
            ("Buffer Size:", "buffer_size", QSpinBox, (64, 1024, 256))
        ]
        
        for label, attr_name, widget_class, values in params:
            widget = widget_class()
            widget.setRange(values[0], values[1])
            widget.setValue(values[2])
            setattr(self, attr_name, widget)
            self.proc_layout.addRow(label, widget)
        
        self.use_filter = QCheckBox()
        self.use_filter.setChecked(True)
        self.use_filter.setStyleSheet("""
            QCheckBox::indicator:checked {
                background-color: #2ecc71;
                border: 2px solid #27ae60;
            }
        """)
        
        self.proc_layout.addRow("Use Filter:", self.use_filter)
        
        # self.use_bbox = QCheckBox()
        # self.use_bbox.setChecked(True)
        # self.use_bbox.setStyleSheet("""
        #     QCheckBox::indicator:checked {
        #         background-color: #2ecc71;
        #         border: 2px solid #27ae60;
        #     }
        # """)
        
        # self.proc_layout.addRow("Use Bbox:", self.use_bbox)
        
        proc_group.setLayout(self.proc_layout)
        parent_layout.addWidget(proc_group)

    def create_camera_section(self, parent_layout):
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout()
        
        self.resolutions = ["320x320", "480x480", "640x640", "480x640", "640x480", 
                          "800x600", "1024x768", "1280x720", "1280x960", "1440x1080", 
                          "1920x1080", "2560x1440"]
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(self.resolutions)
        self.resolution_combo.setCurrentText("320x320")
        self.resolution_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
                border-left: 2px solid #999;
                border-bottom: 2px solid #999;
                width: 8px;
                height: 8px;
                transform: rotate(-45deg);
                margin-right: 8px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
        """)
        self.resolution_combo.currentTextChanged.connect(self.update_resolution)
        camera_layout.addRow("Resolution:", self.resolution_combo)
        
        camera_group.setLayout(camera_layout)
        parent_layout.addWidget(camera_group)
        
        self.current_width = 320
        self.current_height = 320

    def create_model_section(self, parent_layout):
        self.model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        self.yolo_conf = QDoubleSpinBox()
        self.yolo_conf.setRange(0.1, 1.0)
        self.yolo_conf.setValue(0.6)
        self.yolo_conf.setSingleStep(0.1)
        model_layout.addRow("Confidence:", self.yolo_conf)
        
        self.yolo_iou = QDoubleSpinBox()
        self.yolo_iou.setRange(0.1, 1.0)
        self.yolo_iou.setValue(0.5)
        self.yolo_iou.setSingleStep(0.1)
        model_layout.addRow("IOU:", self.yolo_iou)
        
        self.model_group.setLayout(model_layout)
        parent_layout.addWidget(self.model_group)

    def create_performance_section(self, parent_layout):
        self.perf_group = QGroupBox("Performance Metrics")
        perf_layout = QFormLayout()
        
        self.show_fps = QCheckBox()
        self.show_fps.setChecked(False)
        perf_layout.addRow("Show FPS:", self.show_fps)
        
        self.show_timing = QCheckBox()
        self.show_timing.setChecked(False)
        perf_layout.addRow("Show Timing:", self.show_timing)
        
        self.perf_group.setLayout(perf_layout)
        parent_layout.addWidget(self.perf_group)

    def create_control_section(self, parent_layout):
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        self.start_button = QPushButton("▶ Start")
        self.start_button.clicked.connect(self.start_processing)
        
        self.stop_button = QPushButton("⬛ Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
                
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.status_label)
        
        control_group.setLayout(control_layout)
        parent_layout.addWidget(control_group)

    def create_display_section(self, parent_layout):
        display_group = QGroupBox("Video Display")
        display_layout = QVBoxLayout()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar {
                background: #f0f0f0;
            }
            QScrollBar:horizontal {
                height: 12px;
            }
            QScrollBar:vertical {
                width: 12px;
            }
        """)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        container_layout.addWidget(self.display_label)
        scroll_area.setWidget(container)
        display_layout.addWidget(scroll_area)
        display_group.setLayout(display_layout)
        parent_layout.addWidget(display_group)

    def update_resolution(self, resolution_text):
        width, height = map(int, resolution_text.split('x'))
        self.current_width = width
        self.current_height = height

    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input", "", "Event Files (*.raw *.dat)")
        if file_path:
            self.input_path.setText(file_path)

    def browse_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Output", "", "Video Files (*.mp4)")
        if file_path and not file_path.endswith('.mp4'):
            file_path += '.mp4'
        if file_path:
            self.output_path.setText(file_path)
            
    def validate_paths(self):
        """Validate input and output paths before processing"""
        # If using camera, skip input validation
        if self.camera_radio.isChecked():
            # Only validate output path if provided
            if self.output_path.text():
                if not self.output_path.text().lower().endswith(('.mp4', '.avi')):
                    QMessageBox.warning(
                        self,
                        "Invalid Output",
                        "Output file must be .mp4 or .avi format.",
                        QMessageBox.Ok
                    )
                    return False
                
                # Check if output directory exists
                output_dir = os.path.dirname(self.output_path.text())
                if output_dir and not os.path.exists(output_dir):
                    QMessageBox.warning(
                        self,
                        "Invalid Output",
                        "Output directory does not exist.",
                        QMessageBox.Ok
                    )
                    return False
            return True
        
        # File mode validation
        input_path = self.input_path.text()
        
        # Validate input file
        if not input_path:
            QMessageBox.warning(
                self,
                "Missing Input",
                "Please select an input file.",
                QMessageBox.Ok
            )
            return False
            
        if not os.path.exists(input_path):
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Input file does not exist.",
                QMessageBox.Ok
            )
            return False
            
        if not input_path.lower().endswith(('.raw', '.dat')):
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Input file must be a .raw or .dat file.",
                QMessageBox.Ok
            )
            return False
        
        return True

    def start_processing(self):
        self.loading_dialog = LoadingDialog(
            self, 
            "Starting processing...", 
            duration=6000  
        )
        self.loading_dialog.show()

        loop = QEventLoop()
        
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: (
            restart_usb_devices(),
            loop.quit()
        ))
        timer.start(10)  
        
        loop.exec_()

        if not self.validate_paths():
            return
        
        # Prepare processing parameters
        params = ProcessingParams(
            neighborhood_size=self.neighborhood_size.value(),
            time_tolerance=self.time_tolerance.value(),
            buffer_size=self.buffer_size.value(),
            use_filter=self.use_filter.isChecked(),
            output_height=self.output_height.value(),
            show_fps=self.show_fps.isChecked(),
            show_timing=self.show_timing.isChecked(),
            yolo_conf=self.yolo_conf.value(),
            yolo_iou=self.yolo_iou.value(),
            camera_width=self.current_width,
            camera_height=self.current_height
        )
        
        output_path = self.output_path.text().strip()
        
        # Create processing thread
        self.processing_thread = ProcessingThread(self.base_path,
            self.config,
            "" if self.camera_radio.isChecked() else self.input_path.text(),
            output_path, 
            params
        )
        
        # Connect signals
        self.processing_thread.frame_ready.connect(self.update_display)
        self.processing_thread.finished.connect(self.processing_finished)
        
        # Start thread
        self.processing_thread.start()
        
        # Disable buttons
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Processing...")

    def stop_processing(self):
        self.loading_dialog = LoadingDialog(
            self, 
            "Stopping processing...", 
            duration=3000 
        )
        self.loading_dialog.show()

        loop = QEventLoop()
        
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: (
            self._stop_processing_internal(),
            loop.quit()
        ))
        timer.start(10)  
        
        loop.exec_()

    def _stop_processing_internal(self):
        if self.processing_thread and self.processing_thread.isRunning():
            # Stop processing
            self.processing_thread.running = False
            
            self.processing_thread.wait(5000) 
            
            if self.processing_thread.isRunning():
                print("Forcefully terminating processing thread")
                self.processing_thread.terminate()
            
            # Stop stream processor
            if hasattr(self.processing_thread, 'stream_processor'):
                self.processing_thread.stream_processor.stop()
            
            # Reset UI
            self.status_label.setText("Stopped")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def processing_finished(self, success, message):
        if hasattr(self, 'loading_dialog') and self.loading_dialog:
            pass
        
        # Update UI
        self.status_label.setText(message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Show error message if processing failed
        if not success:
            QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        # Close loading dialog if it's open
        if hasattr(self, 'loading_dialog') and self.loading_dialog:
            self.loading_dialog.close()
        
        # Stop processing
        self.stop_processing()
        event.accept()
        
    def update_display(self, frame):
        try:
            self.frame_buffer = frame.copy()  # Create a copy to avoid memory conflicts
            
            frame = cv2.cvtColor(self.frame_buffer, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            
            # Use a try-except block for QImage creation
            try:
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                    self.display_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.display_label.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"QImage creation error: {e}")
                
        except Exception as e:
            print(f"Frame processing error: {e}")

    def processing_finished(self, success, message):
        self.status_label.setText(message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if not success:
            QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

