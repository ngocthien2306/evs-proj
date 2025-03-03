from typing import List, Optional, Union
import logging
import time
from queue import Queue, Full, Empty
from threading import Event

from src.application.interfaces.counter import ICounter
from src.application.interfaces.detector import IDetector
from src.application.interfaces.tracker import ITracker
from src.application.interfaces.event_iterator import IEventIterator
from src.domain.entities.detection import Frame
from src.domain.entities.detection import Detection
from src.domain.value_objects.counting_zone import CountingZone

class VideoDetectionService:
    def __init__(self,
                 frame_iterator: IEventIterator,
                 detector: IDetector,
                 tracker: ITracker,
                 counter: Optional[ICounter] = None,
                 max_buffer_size: int = 1):
        
        self.frame_iterator = frame_iterator
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.frame_queue = Queue(maxsize=max_buffer_size)
        self.running = Event()
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0
        
    def process_frames_thread(self) -> None:
        """Thread function to read frames from video source"""
        self.running.set()
        try:
            for frame_data in self.frame_iterator:
                if not self.running.is_set():
                    break

                try:
                    frame = Frame(
                        data=frame_data,
                        height=frame_data.shape[0],
                        width=frame_data.shape[1]
                    )
                    self.frame_queue.put(frame, timeout=0.1)
                except Full:
                    logging.warning("Frame queue is full, skipping frame")
                    continue

        except Exception as e:
            logging.error(f"Error in process_frames_thread: {e}")
        finally:
            self.running.clear()

    def process_detections(self, counting_zones: Optional[List[CountingZone]] = None):
        """Main processing loop for detections"""
        self.running.set()
        try:
            while self.running.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue

                self._update_fps()
                frame.fps = self._current_fps

                # Detect objects
                detections = self.detector.detect(frame)
                
                # Update tracking and counting based on counter type
                if self.counter:
                    if hasattr(self.counter, 'update_tracks'):  # Line counter
                        tracked_detections = self.tracker.update(detections)
                        frame.detections = tracked_detections
                        if counting_zones:
                            counting_zones = self.counter.update(tracked_detections, counting_zones)
                        yield frame, counting_zones
                    else:  # Frame counter
                        frame.detections = detections
                        counting_result = self.counter.update(detections)
                        yield frame, counting_result
                else:
                    frame.detections = detections
                    yield frame, None

                self.frame_queue.task_done()

        except Exception as e:
            logging.error(f"Error in process_detections: {e}")
        finally:
            self.running.clear()

    def _update_fps(self) -> None:
        """Update FPS calculation"""
        self._frame_count += 1
        if time.time() - self._fps_start_time >= 1.0:
            self._current_fps = self._frame_count
            self._frame_count = 0
            self._fps_start_time = time.time()