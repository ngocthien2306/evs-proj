from abc import ABC, abstractmethod
from typing import List
from ...domain.entities.detection import Detection, Frame

class IDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        """Perform detection on a frame"""
        pass