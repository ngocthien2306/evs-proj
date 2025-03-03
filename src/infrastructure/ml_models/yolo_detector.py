from typing import List
import numpy as np
from ultralytics import YOLO
from ...domain.entities.detection import Detection, BoundingBox, Keypoints, Frame
from ...application.interfaces.detector import IDetector

class YoloDetector(IDetector):
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: Frame) -> List[Detection]:
        results = self.model.predict(
            frame.data, 
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        detections = []
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls_id = r
            bbox = BoundingBox(x1, y1, x2, y2, conf, int(cls_id))
            
            keypoints = None
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                kpts = results.keypoints[0].data
                points = [(float(x), float(y)) for x, y, conf in kpts]
                confidences = [float(conf) for _, _, conf in kpts]
                keypoints = Keypoints(points, confidences)
                
            detections.append(Detection(bbox=bbox, keypoints=keypoints))
            
        return detections