import multiprocessing
import os
import sys
import time
from PyQt5.QtWidgets import QApplication
from src.infrastructure.utils.system_utils import restart_usb_devices
from src.interface.gui.gui import EventProcessorGUI

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    
    base_path = get_base_path()

    
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    
    if os.getpid() == multiprocessing.current_process().pid:
        app = QApplication(sys.argv)
        restart_usb_devices()
        time.sleep(2)
        window = EventProcessorGUI(base_path)
        window.show()
        sys.exit(app.exec_())