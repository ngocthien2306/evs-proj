import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None):
    """
    Thiết lập logger với định dạng và handlers phù hợp.
    
    Args:
        name (str): Tên của logger
        log_file (str, optional): Đường dẫn file log. Nếu None, sẽ tạo file theo timestamp
    """
    if log_file is None:
        # Tạo thư mục logs nếu chưa tồn tại
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/event_radar_{timestamp}.log'

    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Tạo file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Tạo console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Định dạng log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Thêm handlers vào logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger 