from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

@dataclass
class EventFrame:
    data: np.ndarray
    original_dimensions: Tuple[int, int]  # (width, height)
    crop_coordinates: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    events: np.ndarray = None