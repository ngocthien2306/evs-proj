from typing import List, Optional, Union
import logging
import time
from queue import Queue, Full, Empty
from threading import Event

from src.infrastructure.counting.frame_counter import FrameCounter
from src.infrastructure.counting.line_counter import LineCounter
from src.application.interfaces.counter import ICounter
from src.application.interfaces.detector import IDetector
from src.application.interfaces.tracker import ITracker
from src.application.interfaces.event_iterator import IEventIterator
from src.domain.entities.detection import Frame
from src.infrastructure.event_camera.processors.event_processor import EventProcessor
from src.domain.entities.detection import Detection
from src.domain.value_objects.counting_zone import CountingZone


class DetectionService:
    def __init__(self,
                 event_iterator: IEventIterator,
                 detector: IDetector,
                 tracker: ITracker,
                 counter: Optional[Union[LineCounter, FrameCounter]] = None,
                 event_processor: Optional[EventProcessor] = None,
                 max_buffer_size: int = 1):
        
        self.event_iterator = event_iterator
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.event_processor = event_processor or EventProcessor()
        self.image_queue = Queue(maxsize=max_buffer_size)
        self.running = Event()
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0
        
    def process_events(self) -> None:
        self.running.set()
        try:
            for events in self.event_iterator:
                if not self.running.is_set():
                    break

                # Process events into frame
                event_frame = self.event_processor.create_frame(events)
                
                try:
                    frame = Frame(
                        data=event_frame.data,
                        height=event_frame.data.shape[0],
                        width=event_frame.data.shape[1]
                    )
                    self.image_queue.put(frame, timeout=0.1)
                except Full:
                    logging.warning("Image queue is full, skipping frame.")
                    continue

        except Exception as e:
            logging.error(f"Error in process_events: {e}")
        finally:
            self.running.clear()

    def process_frames(self, counting_zones: Optional[List[CountingZone]] = None) -> None:
        self.running.set()
        try:
            while self.running.is_set():
                try:
                    frame = self.image_queue.get(timeout=0.001)
                except Empty:
                    continue

                self._update_fps()
                frame.fps = self._current_fps

                # Detect objects
                detections = self.detector.detect(frame)
                
                if isinstance(self.counter, LineCounter):
                    tracked_detections = self.tracker.update(detections)
                    frame.detections = tracked_detections
                    
                    if counting_zones:
                        counting_zones = self.counter.update(
                            tracked_detections, 
                            counting_zones
                        )
                    yield frame, counting_zones
                    
                elif isinstance(self.counter, FrameCounter):
                    frame.detections = detections
                    
                    counting_result = self.counter.update(detections)
                    yield frame, counting_result
                    
                else:
                    frame.detections = detections
                    yield frame, None

                self.image_queue.task_done()

        except Exception as e:
            logging.error(f"Error in process_frames: {e}")
        finally:
            self.running.clear()

    def _update_fps(self) -> None:
        self._frame_count += 1
        if time.time() - self._fps_start_time >= 1.0:
            self._current_fps = self._frame_count
            self._frame_count = 0
            self._fps_start_time = time.time()
            