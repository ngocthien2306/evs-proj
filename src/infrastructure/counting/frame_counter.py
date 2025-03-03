from collections import deque
from typing import List
from src.application.interfaces.counter import ICounter
from src.domain.entities.detection import Detection
from src.domain.value_objects.counting_result import CountingResult


class FrameCounter(ICounter):
    def __init__(self, 
                 alpha: float = 0.3, 
                 count_threshold: int = 2,
                 temporal_window: int = 5):
        self.counting_result = CountingResult()
        self.alpha = alpha
        self.ema_count = 0
        self.is_initialized = False
        self.count_threshold = count_threshold
        self.temporal_window = temporal_window
        self.recent_counts = deque(maxlen=temporal_window)

    def update(self, detections: List[Detection]) -> CountingResult:
        # Lọc detections có confidence cao
        reliable_detections = [
            det for det in detections 
            if det.bbox.confidence >= self.count_threshold
        ]
        
        current_count = len(reliable_detections)
        self.recent_counts.append(current_count)

        if len(self.recent_counts) == self.temporal_window:
            avg_count = sum(self.recent_counts) / len(self.recent_counts)
            count_diff = abs(avg_count - self.ema_count)
            
            if count_diff >= self.count_threshold:
                if not self.is_initialized:
                    self.ema_count = avg_count
                    self.is_initialized = True
                else:
                    self.ema_count = (self.alpha * avg_count + 
                                    (1 - self.alpha) * self.ema_count)

        smoothed_count = int(round(self.ema_count))
        
        self.counting_result.current_count = current_count
        self.counting_result.smoothed_count = smoothed_count
        
        if smoothed_count > self.counting_result.max_count:
            self.counting_result.max_count = smoothed_count
            
        return self.counting_result