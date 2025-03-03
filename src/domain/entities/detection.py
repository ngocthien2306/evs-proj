from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

@dataclass
class Keypoints:
    points: List[Tuple[float, float]]
    confidences: List[float]

@dataclass
class Detection:
    bbox: BoundingBox
    track_id: Optional[int] = None
    keypoints: Optional[Keypoints] = None

@dataclass
class Frame:
    data: np.ndarray
    height: int
    width: int
    fps: float = 0.0
    detections: List[Detection] = None