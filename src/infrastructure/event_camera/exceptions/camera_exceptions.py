
class CameraException(Exception):
    """Base exception for camera-related errors"""
    pass

class CameraNotFoundError(CameraException):
    """Raised when camera device cannot be found"""
    pass

class InvalidFileError(CameraException):
    """Raised when input file is invalid"""
    pass