from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class CountingZone:
    points: List[Tuple[int, int]]  # Polygon points defining the counting zone
    name: str
    in_count: int = 0
    out_count: int = 0
