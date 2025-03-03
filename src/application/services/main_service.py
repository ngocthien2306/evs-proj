import multiprocessing
import cv2
import time
from collections import deque
from queue import Empty
from multiprocessing import Process, Queue, Event
from src.infrastructure.utils.module_process import (all_one_thread, event_process, 
                                                     fall_detection_process, people_detection_process_onnx, 
                                                     person_direction_process, visualization_process)
from src.domain.value_objects.config import RadarConfig
from src.domain.entities.entities import ProcessingParams
import os
       
class LogicFactoryService:
    def __init__(self, base_path="", frame_queue_size=5, resize_dims=(320, 320), target_fps=30, input_filename="", args: ProcessingParams = None):
        # Queues for inter-process communications
        self.base_path = base_path
        self.yolo_path = os.path.join(base_path, "public", "yolo_models", "best_s_22_1.onnx")
        self.tflite_path = os.path.join(base_path, "public", "tflite_models", "FE.V03.D05.tflite")
        self.fall_model_path = os.path.join(base_path, "public", "onnx_models", "WQ_TiVi.onnx")
        self.config_path = os.path.join(base_path, "public", "bias", 'bias.bias')
        
        self.frame_queue = Queue(maxsize=frame_queue_size)
        self.frame_queue_fall = Queue(maxsize=64)
        self.event_queue = Queue(maxsize=frame_queue_size)
        self.people_queue = Queue(maxsize=frame_queue_size)
        self.fall_queue = Queue(maxsize=frame_queue_size)
        self.result_queue = Queue(maxsize=frame_queue_size)
        self.radar_queue = Queue(maxsize=frame_queue_size)
        self.result_queue_single = Queue(maxsize=256)
        self.running = Event()
        self.config = RadarConfig()
        self.args = args
        
        self.frame_height = args.output_height
        self.camera_resolution = (args.camera_width, args.camera_height)
        
        self.resize_dims = resize_dims
        self.target_fps = target_fps
        self.input_filename = input_filename
        
        # Stats
        self.fps_buffer = deque(maxlen=30)
        self.people_buffer = deque(maxlen=5)
        self.prev_time = time.time()
        
    def update_people_count(self, count):
        self.people_buffer.append(count)
        counts = list(self.people_buffer)
        return max(set(counts), key=counts.count) if counts else 0

    def draw_stats(self, frame, fps, people_count, fall_status):
        cv2.rectangle(frame, (10, 2), (200, 85), (0, 0, 0), -1)
        cv2.putText(frame, f'FPS: {fps:.1f}', (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'People: {people_count}', (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        color = (0, 0, 255) if fall_status == 'FALL' else (0, 255, 0)
        cv2.putText(frame, f'Status: {fall_status}', (15, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
    
    def start_ui_one_thread(self):
        self.all_process = Process(
            name="AllProcess",
            target=all_one_thread, 
            args=(self.result_queue_single, self.input_filename, self.camera_resolution, self.yolo_path, self.args, self.frame_height, self.config_path, self.fall_model_path),
            daemon=True
        )
        self.all_process.start()
        return self.result_queue_single
        
    def start_ui(self):
        """Start processing for UI display without OpenCV window"""
        self.running.set()
        
        # Start all processes
        self.event_process = Process(
            name="EventProcess",
            target=event_process, 
            args=(self.frame_queue, self.frame_queue_fall, self.event_queue, self.running, self.input_filename, self.camera_resolution, self.config_path),
            daemon=True
        )
        
        self.people_process = Process(
            name="PeopleProcess",
            target=people_detection_process_onnx, 
            args=(self.frame_queue, self.people_queue, self.running, self.yolo_path, self.args),
            daemon=True
        )
        
        self.fall_process = Process(
            name="FallProcess",
            target=fall_detection_process,
            args=(self.frame_queue, self.fall_queue, self.running, self.fall_model_path),
            daemon=True
        )
        
        self.person_direction = Process(
            name="PersonDirection",
            target=person_direction_process,
            args=(self.event_queue , self.radar_queue, self.running, self.args),
            daemon=True
        )
        
        self.viz_process = Process(
            name="VizProcess",
            target=visualization_process,
            args=(self.people_queue, self.fall_queue, self.radar_queue, self.result_queue, self.running, self.frame_height, self.args),
            daemon=True
        )
        
        self.event_process.start()
        self.person_direction.start()
        self.people_process.start()
        self.fall_process.start()
        self.viz_process.start()

        # Return result_queue to ProcessingThread can be get frame
        return self.result_queue


    def stop(self):
        print("Stopping all processes...")
        self.running.clear()
        
        with multiprocessing.Pool() as pool:
            pool.terminate()
            pool.join()
        
        for queue in [self.frame_queue, self.people_queue, self.fall_queue, 
                    self.result_queue, self.radar_queue]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
        
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)


