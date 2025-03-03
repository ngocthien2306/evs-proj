import cv2
import numpy as np
import onnxruntime as ort
from src.domain.entities.entities import ProcessingParams
    
class ONNXDetector:
    def __init__(self, model_path: str, params: ProcessingParams):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.params = params
        
        # Get model input details
        model_inputs = self.session.get_inputs()
        self.input_shape = model_inputs[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]

    def preprocess(self, img):
        self.img_height, self.img_width = img.shape[:2]
        
        # Convert BGR to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        
        # Normalize and transpose
        img = np.array(img) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img

    def postprocess(self, img, outputs):
        outputs = np.transpose(np.squeeze(outputs[0]))
        rows = outputs.shape[0]

        boxes = []
        scores = []
        
        # Scale factors
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        for i in range(rows):
            score = outputs[i][4]  # Person score
            
            if score >= self.params.yolo_conf:
                x, y, w, h = outputs[i][0:4]
                
                # Convert to corner format
                x1 = (x - w/2) * x_factor
                y1 = (y - h/2) * y_factor
                x2 = (x + w/2) * x_factor
                y2 = (y + h/2) * y_factor
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)

        # Apply NMS
        if boxes:
            boxes = np.array(boxes)
            scores = np.array(scores)
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                scores.tolist(), 
                self.params.yolo_conf, 
                self.params.yolo_iou
            )
            
            # Convert to normalized coordinates
            boxes_n = boxes[indices].copy()
            boxes_n[:, [0, 2]] /= self.img_width
            boxes_n[:, [1, 3]] /= self.img_height
            
            return boxes[indices], boxes_n, len(indices)
        
        return np.array([]), np.array([]), 0

    def detect_and_track(self, frame):
        # Preprocess
        img_data = self.preprocess(frame)
        
        # Inference
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        
        # Postprocess
        boxes, boxes_n, current_count = self.postprocess(frame, outputs)
        
        return boxes, boxes_n, current_count