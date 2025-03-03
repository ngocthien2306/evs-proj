from typing import Iterator, Optional
import cv2
from src.application.interfaces.event_iterator import IEventIterator
import numpy as np

class VideoFrameIterator(IEventIterator):
    """Iterator class for reading frames from video files or camera streams"""
    
    def __init__(self, 
                 input_source: Optional[str] = None,
                 target_fps: Optional[float] = None,
                 resize_dims: Optional[tuple] = None):
        """
        Initialize video frame iterator
        
        Args:
            input_source: Path to video file or camera index (0 for default camera)
            target_fps: Target frame rate to process video at (None for original video fps)
            resize_dims: Tuple of (width, height) to resize frames to (None for original size)
        """
        # Handle camera index (int) or video file path (str)
        self.input_source = 0 if input_source is None else input_source
        self.cap = cv2.VideoCapture(self.input_source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {input_source}")
            
        # Get video properties
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("FPS of webcam: ", self.original_fps)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set target properties
        self.target_fps = self.original_fps if target_fps is None else target_fps
        self.resize_dims = resize_dims
        
        # Calculate frame skip based on target FPS
        if self.target_fps and self.target_fps < self.original_fps:
            self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        else:
            self.frame_skip = 1
            
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over video frames"""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            # Skip frames based on target FPS
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue
                
            # Resize frame if dimensions specified
            if self.resize_dims:
                frame = cv2.resize(frame, self.resize_dims)
                
            yield frame
            
    def __del__(self):
        """Release video capture resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
            
    def get_size(self) -> int:
        """Get total number of frames in video"""
        return self.frame_count
        
    def get_frame_size(self) -> tuple:
        """Get frame dimensions (width, height)"""
        if self.resize_dims:
            return self.resize_dims
        return (self.width, self.height)
        
    def get_fps(self) -> float:
        """Get effective frame rate"""
        return min(self.target_fps, self.original_fps) if self.target_fps else self.original_fps