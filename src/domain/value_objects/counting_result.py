from dataclasses import dataclass
from collections import deque

@dataclass
class CountingResult:
    current_count: int = 0
    smoothed_count: int = 0
    max_count: int = 0