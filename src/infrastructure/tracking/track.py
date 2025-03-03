from typing import Optional
import numpy as np
from src.domain.entities.detection import Detection, BoundingBox
from .kalman_filter import KalmanFilter

class Track:
    """Track class for managing single object track"""
    def __init__(self, detection: Detection, track_id: int):
        self.track_id = track_id
        self.kalman = KalmanFilter()
        
        # Initialize Kalman filter state
        bbox = detection.bbox
        measurement = np.array([
            bbox.x1,
            bbox.y1,
            bbox.x2 - bbox.x1,  # width
            bbox.y2 - bbox.y1   # height
        ])
        self.kalman.initialize(measurement)
        
        self.time_since_update = 0
        self.hits = 1
        self.detection = detection
        self.confidence = detection.bbox.confidence

    def predict(self) -> Optional[BoundingBox]:
        """Predict next state"""
        prediction = self.kalman.predict()
        if prediction is None:
            return None

        x, y, w, h = prediction
        return BoundingBox(
            x1=float(x),
            y1=float(y),
            x2=float(x + w),
            y2=float(y + h),
            confidence=self.confidence,
            class_id=self.detection.bbox.class_id
        )

    def update(self, detection: Detection) -> None:
        """Update track with detected box"""
        self.detection = detection
        self.confidence = detection.bbox.confidence
        
        bbox = detection.bbox
        measurement = np.array([
            bbox.x1,
            bbox.y1,
            bbox.x2 - bbox.x1,
            bbox.y2 - bbox.y1
        ])
        
        self.kalman.update(measurement)
        self.hits += 1
        self.time_since_update = 0