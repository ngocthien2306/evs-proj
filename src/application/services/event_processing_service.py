from typing import Optional, Tuple
import numpy as np
from ...domain.value_objects.event_frame import EventFrame
from ...infrastructure.event_camera.processors.event_processor import EventProcessor

class EventProcessingService:
    def __init__(self, 
                 width: int = 346,
                 height: int = 260,
                 crop_coordinates: Optional[Tuple[int, int, int, int]] = None):
        self.processor = EventProcessor(
            width=width,
            height=height,
            crop_coordinates=crop_coordinates
        )

    def process_events(self, events: np.ndarray) -> EventFrame:
        """
        Process events into an image frame
        
        Args:
            events: Numpy array of events
            
        Returns:
            EventFrame containing the processed image
        """
        return self.processor.create_frame(events)

    def reset(self) -> None:
        """Reset the event processor's internal state"""
        self.processor.reset_buffer()