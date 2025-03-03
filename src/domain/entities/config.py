from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InputConfig:
    type: str  # "video" or "event"
    file_path: str
    width: int
    height: int
    
    # Video specific settings
    target_fps: Optional[float] = None
    
    # Event camera specific settings
    bias_file: Optional[str] = None
    delta_t: Optional[int] = 10000
    crop_coordinates: Optional[List[int]] = None

@dataclass
class ModelConfig:
    path: str
    confidence_threshold: float

@dataclass
class TrackingConfig:
    max_age: int
    min_hits: int
    iou_threshold: float

@dataclass
class CountingZoneConfig:
    name: str
    points: List[List[int]]

@dataclass
class CountingConfig:
    type: str  # "line" or "frame"
    zones: List[CountingZoneConfig]
    alpha: Optional[float] = 0.3
    count_threshold: Optional[float] = 2
    temporal_window: Optional[int] = 5

@dataclass
class VisualizationConfig:
    track_history_length: int
    display_fps: bool
    display_tracks: bool

@dataclass
class SystemConfig:
    input: InputConfig
    model: ModelConfig
    tracking: TrackingConfig
    visualization: VisualizationConfig
    counting: CountingConfig
