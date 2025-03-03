from typing import List, Tuple
import numpy as np
from shapely.geometry import Point, Polygon
from ...domain.entities.detection import BoundingBox, Detection
from ...domain.value_objects.counting_zone import CountingZone
from ...application.interfaces.counter import ICounter

class LineCounter(ICounter):
    def __init__(self):
        self.previous_positions = {}  # track_id -> previous_position

    def update(self, tracks: List[Detection], zones: List[CountingZone]) -> List[CountingZone]:
        current_positions = {
            t.track_id: self._get_center(t.bbox)
            for t in tracks if t.track_id is not None
        }
        
        for zone in zones:
            polygon = Polygon(zone.points)
            
            for track_id, current_pos in current_positions.items():
                if track_id in self.previous_positions:
                    prev_pos = self.previous_positions[track_id]
                    current_point = Point(current_pos)
                    prev_point = Point(prev_pos)
                    
                    # Check if crossing happened
                    if polygon.contains(current_point) != polygon.contains(prev_point):
                        if polygon.contains(current_point):
                            zone.in_count += 1
                        else:
                            zone.out_count += 1
        
        self.previous_positions = current_positions
        return zones
    
    def _get_center(self, bbox: BoundingBox) -> Tuple[float, float]:
        return ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)