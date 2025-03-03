from abc import ABC, abstractmethod
from typing import List
from ...domain.entities.detection import Detection
from ...domain.value_objects.counting_zone import CountingZone

class ICounter(ABC):
    @abstractmethod
    def update(self, tracks: List[Detection], zones: List[CountingZone]) -> List[CountingZone]:
        """Update counting based on tracks and zones"""
        pass