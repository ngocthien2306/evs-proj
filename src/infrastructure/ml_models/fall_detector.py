from collections import deque
import time

import cv2
import numpy as np

from src.infrastructure.ml_models.evs_fall import evs_fall_exist


class PeopleFall:
    def __init__(self, fall_model_path,  model_type='onnx', buffer_size=60):
        """Initialize PeopleFall

        Args:
            fall_model: Fall detection model instance.
            buffer_size: Size of the buffer for smoothing fall and existence detection.
        """
        self.fall_model_path = fall_model_path
        self.Frame_Array = deque(maxlen=buffer_size)
        self.idx_a = np.array([ii for ii in range(buffer_size) if ii % 2 == 0])

        # Placeholder for the fall detector instance
        self.model_type = model_type
        self.evs_fall_detector = evs_fall_exist(self.fall_model_path,self.model_type, num_threads=1)

        # Buffers for fall and existence smoothing
        self.FALL_Arr = deque([False] * 10, maxlen=10)
        self.EXIST_Arr = deque([False] * 10, maxlen=10)

    def process_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.Frame_Array.append(gray_frame)
        if len(self.Frame_Array) == 60:
            start_time = time.time()
            
            Frame_list = list(self.Frame_Array)
           
            # Run the fall detection model
            output_fall, output_exists = self.evs_fall_detector(Frame_list, self.idx_a)
            # print(output_fall)
            pred_fall = np.squeeze(output_fall)
            pred_exists = np.squeeze(output_exists)

            # Update fall buffer
            if pred_fall >= 0.5:
                class_action = 'FALL'
                self.FALL_Arr.append(True)
            else:
                class_action = 'Normal'
                self.FALL_Arr.append(False)

            # Update existence buffer
            if pred_exists >= 0.5:
                class_exists = 'Person'
                self.EXIST_Arr.append(True)
            else:
                class_exists = 'No Person'
                self.EXIST_Arr.append(False)

            # Determine if a person exists
            class_person_exists = 'Person' if self.EXIST_Arr.count(True) > 5 else ''

            # print('Num of Fall: ', self.FALL_Arr.count(True))
            if self.FALL_Arr.count(True) >= 5:
                param = '*** FALL detected!  sending message.'
                
                self.FALL_Arr = deque([False] * 10, maxlen=10)  # Reset fall buffer
                self.Frame_Array.clear()  # Clear frame buffer
            else:
                self.Frame_Array = deque(Frame_list[5:], maxlen=60)
               
            end_time = time.time()
            print("Fall detect time: ", (round((end_time - start_time) * 1000, 2)))
        
            return class_action
        else:
            return "Normal"