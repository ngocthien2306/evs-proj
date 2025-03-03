from abc import ABC, abstractmethod
from typing import List
from ...domain.entities.detection import Detection

class ITracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracks with new detections"""
        pass