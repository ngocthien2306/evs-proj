from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class CameraBias:
    name: str
    value: int
    limits: Tuple[int, int]

    def increase(self, step: int = 1) -> int:
        new_value = min(self.value + step, self.limits[1])
        self.value = new_value
        return new_value

    def decrease(self, step: int = 1) -> int:
        new_value = max(self.value - step, self.limits[0])
        self.value = new_value
        return new_value