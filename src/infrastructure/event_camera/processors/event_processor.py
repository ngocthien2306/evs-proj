import numpy as np
from typing import Optional, Union, Tuple
from src.domain.value_objects.event_frame import EventFrame

class EventProcessor:
    """Handles conversion of events to image frames"""
    
    def __init__(self, 
                 width: int = 320, 
                 height: int = 320,
                 crop_coordinates: Optional[Tuple[int, int, int, int]] = (225, 50, 720, 600)):
        self.width = width
        self.height = height
        self.crop_coordinates = crop_coordinates
        self._buffer = None if crop_coordinates else np.ones((height, width, 3), dtype=np.uint8) * 127

    def create_frame(self, 
                    events: np.ndarray, 
                    reuse_buffer: bool = True) -> EventFrame:
        """
        Convert events to a binary histogram image frame.
        
        Args:
            events: Structured numpy array with fields ('x', 'y', 'p', 't')
            reuse_buffer: Whether to reuse the existing image buffer
        
        Returns:
            EventFrame object containing the processed image
        """
        # Convert events to correct dtype if needed
        if events.dtype != [('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')]:
            events = np.array(events, 
                            dtype=[('x', '<u2'), ('y', '<u2'), 
                                  ('p', '<i2'), ('t', '<i8')])

        # Create or reuse image buffer
        if self._buffer is None or not reuse_buffer:
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 127
        else:
            img = self._buffer
            img[...] = 127

        if events.size:
            assert events['x'].max() < self.width, "out of bound events: x = {}, w = {}".format(events['x'].max(), self.width)
            assert events['y'].max() < self.height, "out of bound events: y = {}, h = {}".format(events['y'].max(), self.height)

            img[events['y'], events['x'], :] = 255 * events['p'][:, None]

        # Apply cropping if specified
        if self.crop_coordinates:
            x1, y1, x2, y2 = self.crop_coordinates
            img = img[y1:y2, x1:x2]

        # Store buffer for reuse if needed
        if reuse_buffer:
            self._buffer = img

        return EventFrame(
            data=img,
            original_dimensions=(self.width, self.height),
            crop_coordinates=self.crop_coordinates,
            events=events
        )

    def reset_buffer(self) -> None:
        """Reset the internal image buffer"""
        if self._buffer is not None:
            self._buffer[...] = 127



