import logging
import multiprocessing
import signal
import traceback
import cv2
import time
import numpy as np
from collections import deque
from queue import Full, Empty
from multiprocessing import Process, Queue, Event

from src.domain.value_objects.config import RadarConfig
from src.domain.entities.entities import ProcessingParams
from src.core.filters import ActivityNoiseFilterGPU, BoundingBox, filter_events
from src.core.radar import ActivityMonitor, RadarViewer
from src.infrastructure.ml_models.evs_fall import evs_fall_exist
from src.infrastructure.ml_models.onnx_detect import ONNXDetector
from src.infrastructure.event_camera.iterators.bias_events_iterator import BiasEventsIterator
from src.infrastructure.event_camera.processors.event_processor import EventProcessor
from src.infrastructure.utils.utils import draw_radar_background, make_binary_histo
import cupy as cp
import os
import sys
from multiprocessing.shared_memory import SharedMemory
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

MAX_BBOXES = 3
SENTINEL = -1.0

def get_bbox_shm():
    shm_name = "bbox_shm"
    size = MAX_BBOXES * 4 * 5
    try:
        shm = SharedMemory(name=shm_name)
    except FileNotFoundError:
        shm = SharedMemory(name=shm_name, create=True, size=size)
    return shm

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

base_path = get_base_path()
yolo_path = os.path.join(base_path, "yolo_models")
tflite_path = os.path.join(base_path, "tflite_models")
onnx_path = os.path.join(base_path, "onnx_models")
config_path = os.path.join(base_path, "configs")

class PeopleFall:
    def __init__(self, model_type='onnx', buffer_size=60):
        """Initialize PeopleFall

        Args:
            fall_model: Fall detection model instance.
            buffer_size: Size of the buffer for smoothing fall and existence detection.
        """
        self.fall_model = os.path.join(onnx_path, 'WQ_TiVi.onnx')
        self.Frame_Array = deque(maxlen=buffer_size)
        self.idx_a = np.array([ii for ii in range(buffer_size) if ii % 2 == 0])

        # Placeholder for the fall detector instance
        self.model_type = model_type
        self.evs_fall_detector = evs_fall_exist(self.fall_model,self.model_type, num_threads=1)

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
      
def restart_usb_devices():
    try:
        import subprocess
        print("Starting Setup ...")
        subprocess.run(['pnputil', '/restart-device', 'USB\\VID*'], 
                      capture_output=True, 
                      text=True, 
                      creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(2)  
        return True
    except Exception as e:
        print(f"Failed to restart USB: {e}")
        return False

def event_process(frame_queue, frame_queue_frame, event_queue , running, input_filename, camera_resolution):
    try:
        w, h = camera_resolution
        processor = EventProcessor(
            width=w,
            height=h,
            crop_coordinates=None
        )
        
        iterator = None if not input_filename else BiasEventsIterator(
            delta_t=33000,
            input_filename=input_filename,
            bias_file=os.path.join(config_path, 'bias.bias')
        )
        
        zero_event_counter = 0
        max_zero_events = 10
        error_counter = 0
        max_errors = 3
        
        while running.is_set():
            try:
                if iterator is None and not input_filename:
                    try:
                        iterator = BiasEventsIterator(
                            delta_t=33000,
                            input_filename=input_filename,
                            bias_file=os.path.join(config_path, 'bias.bias')
                        )
                        error_counter = 0  
                    except Exception as e:
                        print(f"Failed to create iterator: {e}")
                        error_counter += 1
                        if error_counter >= max_errors:
                            restart_usb_devices()
                            error_counter = 0
                        time.sleep(1)
                        continue
                
                for events in iterator:
                    
                    if not running.is_set():
                        break
                    
                    start_time = time.time()
                    
                    event_count = len(events)
                    # print(event_count)

                    if event_count == 0:
                        zero_event_counter += 1
                        if zero_event_counter >= max_zero_events:
                            print("Too many zero events, reloading USB...")
                            if iterator:
                                del iterator
                                iterator = None
                            if restart_usb_devices():
                                print("USB restarted successfully")
                            else:
                                print("Failed to restart USB")
                            zero_event_counter = 0
                            break
                    else:
                        zero_event_counter = 0 
                    if event_queue.full():
                        try:
                            event_queue.get_nowait()
                        except Empty:
                            pass
                    try:
                        event_queue.put((events, time.time()))
                    except Full:
                        pass
                    event_frame = processor.create_frame(events)
                    # cv2.imwrite("event_frame.png", event_frame.data)
                    end_time = time.time()
                    processing_time = round((end_time - start_time) * 1000, 2)
                    
                    
                    
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except Empty:
                            pass

                    try:
                        frame_queue.put((event_frame, time.time(), processing_time))
                    except Full:
                        pass
                    

                    if frame_queue_frame.full():
                        try:
                            frame_queue_frame.get_nowait()
                        except Empty:
                            pass

                    try:
                        frame_queue_frame.put((event_frame, time.time(), processing_time))
                    except Full:
                        pass
                    
                if input_filename:  # File mode
                    print("File processing completed")
                    frame_queue.put(("EOF", time.time()))
                    break
                else:  # Camera mode
                    if iterator:
                        del iterator
                        iterator = None

            except Exception as e:
                print(f"Iterator error: {e}")
                if "TimeHigh discrepancy" in str(e):
                    print("TimeHigh discrepancy detected, reloading USB...")
                    if iterator:
                        del iterator
                        iterator = None
                    restart_usb_devices()
                    continue
                
                error_counter += 1
                if error_counter >= max_errors:
                    print(f"Too many errors ({error_counter}), restarting USB...")
                    restart_usb_devices()
                    error_counter = 0
                
                if iterator:
                    del iterator
                    iterator = None
                
                if not input_filename:  # Chỉ retry với camera mode
                    time.sleep(1)
                else:
                    break

    except Exception as e:
        print(traceback.print_exc())
        logging.error(f"Error in event process: {e}")
    finally:
        if iterator:
            del iterator
        running.clear()

def people_detection_process_onnx(frame_queue, people_queue, running, model_path, args: ProcessingParams):
    """
    Process frames for people detection using ONNX model
    
    Args:
        frame_queue: Queue for input frames
        people_queue: Queue for detection results
        running: Event to control process
        model_path: Path to ONNX model
        args: Processing parameters
    """
    try:
        # Initialize detector
        detector = ONNXDetector(model_path, args)
        last_frame = None
        
        while running.is_set():
            try:
                # Get frame from queue
                event_frame, timestamp, _ = frame_queue.get(timeout=0.1)
                frame = event_frame.data
                last_frame = frame
            except Empty:
                if last_frame is None:
                    continue
                frame = last_frame
                timestamp = time.time()
            
            # Process frame
            start_time = time.time()
            
            # Run detection
            boxes, boxes_n, current_count = detector.detect_and_track(frame)
            
            # Create tracked frame
            tracked_frame = frame.copy()
            
            # Draw detections
            if current_count > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(tracked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(tracked_frame, "Person", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate processing time
            end_time = time.time()
            processing_time = round((end_time - start_time) * 1000, 2)
            
            # cv2.imwrite("tracked_frame.png", tracked_frame)
            
            # Handle queue operations
            if people_queue.full():
                try:
                    people_queue.get_nowait()
                except Empty:
                    pass
            
            try:
                people_queue.put((tracked_frame, boxes_n, current_count, timestamp, processing_time))
            except Full:
                people_queue.put((frame, [], 0, timestamp, 0))
                continue
                
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error in people detection process: {e}")
    finally:
        running.clear()
        
def people_detection_process(frame_queue, people_queue, running, model_path, args: ProcessingParams):
    bbox_shm = get_bbox_shm()
    boxes_shared = np.ndarray((MAX_BBOXES, 4), dtype=np.float32, buffer=bbox_shm.buf)
    
    try:
        # model = YOLO(model_path, task="detect")
        model = None
        last_frame = None
        while running.is_set():
            try:
                event_frame, timestamp, _ = frame_queue.get(timeout=0.1)
                frame = event_frame.data
                last_frame = frame
            except Empty:
                if last_frame is None:
                    continue
                frame = last_frame
                timestamp = time.time()
                
            start_time = time.time()
            results = model.track(
                frame, 
                tracker=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "botsort.yaml"),
                persist=True,
                verbose=False,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                imgsz=320
            )
            
            tracked_frame = frame.copy()
            current_count = 0
            boxes_n = []
            
            if results and results[0].boxes.id is not None:
                boxes_n = results[0].boxes.xyxyn.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                current_count = len(track_ids)
                num_boxes = min(current_count, MAX_BBOXES)
                boxes_shared[:num_boxes] = boxes_n[:num_boxes].astype(np.float32)
                if num_boxes < MAX_BBOXES:
                    boxes_shared[num_boxes:] = SENTINEL
                    
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(tracked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(tracked_frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                boxes_shared[:] = SENTINEL
            end_time = time.time()
            processing_time = round((end_time - start_time) * 1000, 2)
            if people_queue.full():
                try:
                    people_queue.get_nowait()
                except Empty:
                    pass
            try:
                people_queue.put((tracked_frame, boxes_n, current_count, timestamp, processing_time))
            except Full:
                people_queue.put((frame, [], 0, timestamp, 0))
                continue
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error in people detection process: {e}")
    finally:
        running.clear()

# def convert_tracker_boxes(events, normalized_boxes, frame_width: int, frame_height: int, expansion_ratio: float = -0.2, max_peak: int = 3):
#     bb_list = [BoundingBox(det[0], det[1], det[2], det[3], frame_width, frame_height, expansion_ratio)
#                for det in normalized_boxes]
#     if len(normalized_boxes) == 0:
#         return np.empty(0, dtype=events.dtype)
#     x = events['x']
#     y = events['y']
#     if y.dtype != np.int32:
#         y = y.astype(np.int32)
#     valid_mask = (x >= 0) & (x < frame_width) & (y >= 0) & (y < frame_height)
#     if len(normalized_boxes) == 1:
#         det = normalized_boxes[0]
#         bb = BoundingBox(det[0], det[1], det[2], det[3], frame_width, frame_height, expansion_ratio)
#         bbox_mask = (x >= bb.x_min) & (x <= bb.x_max) & (y >= bb.y_min) & (y <= bb.y_max)
#     else:
#         bb_list = [BoundingBox(det[0], det[1], det[2], det[3], frame_width, frame_height, expansion_ratio)
#                    for det in normalized_boxes[:max_peak]]
#         bbox_masks = [(x >= bb.x_min) & (x <= bb.x_max) & (y >= bb.y_min) & (y <= bb.y_max) for bb in bb_list]
#         bbox_mask = np.logical_or.reduce(bbox_masks)
#     combined_mask = valid_mask & bbox_mask
#     filtered_events = events[combined_mask]
#     return filtered_events
def convert_tracker_boxes(events, normalized_boxes, frame_width: int, frame_height: int, expansion_ratio: float = -0.2, max_threads: int = 3):
    bb_list = [BoundingBox(det[0], det[1], det[2], det[3], frame_width, frame_height, expansion_ratio)
               for det in normalized_boxes]
    x = events['x']
    y = events['y'].astype(np.int32)
    print("Bbox length:", len(normalized_boxes))
    masks = [((x >= bb.x_min) & (x <= bb.x_max) & (y >= bb.y_min) & (y <= bb.y_max))
             for bb in bb_list[:max_threads]]
    combined_mask = np.logical_or.reduce(masks)
    x_filtered = x[combined_mask]
    y_filtered = y[combined_mask]
    filtered_size = len(x_filtered)
    valid_indices = (x >= 0) & (x < frame_width) & (y >= 0) & (y < frame_height)
    filtered_events = np.empty(filtered_size, dtype=events.dtype)
    filtered_events['t'] = events['t'][valid_indices][:filtered_size]
    filtered_events['p'] = events['p'][valid_indices][:filtered_size]
    filtered_events['x'] = x_filtered
    filtered_events['y'] = y_filtered
    return filtered_events

def convert_events_data(events: np.ndarray,):
    if events.dtype != [('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')]:
            events = np.array(events, 
                            dtype=[('x', '<u2'), ('y', '<u2'), 
                                  ('p', '<i2'), ('t', '<i8')])
    return events

def person_direction_process(event_queue, radar_queue, running, args: ProcessingParams):
    bbox_shm = get_bbox_shm()
    boxes_shared = np.ndarray((MAX_BBOXES, 4), dtype=np.float32, buffer=bbox_shm.buf)
    last_valid_angles = []
    no_movement_counter = 0
    last_frame_time = time.time()
    last_peak_time = time.time()
    last_bbox = None
    last_bbox_time = 0
    frame_width = args.camera_width
    frame_height = args.camera_height
    radar_height = int(frame_height * 0.95)
    radar_width = int(frame_width * 1.47)
    config = RadarConfig()
    lateral_fov_deg = np.rad2deg(config.lateral_fov)
    pixel_margin = int(config.trim_angle_deg / (lateral_fov_deg / frame_width))
    activity_monitor = ActivityMonitor(config, frame_width)
    radar_viewer = RadarViewer(config, radar_width, radar_height)
    if config.use_filter:
        activity_filter = ActivityNoiseFilterGPU(
            width=frame_width,
            height=frame_height,
            threshold=config.delta_t,
            neighborhood_size=args.neighborhood_size,
            time_tolerance=args.time_tolerance
        )
    else:
        activity_filter = None

    def update_radar_visualization(ev_rate_gpu: cp.ndarray) -> np.ndarray:
        nonlocal last_frame_time, last_valid_angles, last_peak_time
        radar_img = np.copy(radar_viewer.linear_bins_cpu)
        draw_radar_background(radar_img, radar_width, radar_height, config.lateral_fov)
        frame_delta = time.time() - last_frame_time
        last_frame_time = time.perf_counter()
        if no_movement_counter > 10 and last_valid_angles:
            angles_to_show = [float(a + v) for a, v in zip(last_valid_angles, radar_viewer.angle_velocities)]
        else:
            angles_found = radar_viewer.compute_view(ev_rate_gpu, radar_img, frame_delta)
            angles_to_show = radar_viewer.stabilize_angles(angles_found)
            if angles_to_show:
                last_valid_angles = angles_to_show.copy()
                last_peak_time = time.time()  
        if (not angles_to_show or len(angles_to_show) == 0) and (time.time() - last_peak_time > 1):
            radar_viewer.tracked_peaks.clear()
            last_valid_angles = []
            draw_radar_background(radar_img, radar_width, radar_height, config.lateral_fov)
        
        return radar_img

    try:
        last_events = None
        while running.is_set():
            try:
                events_data, timestamp = event_queue.get(timeout=0.1)
                events = convert_events_data(events_data)
                event_image = make_binary_histo(events, width=args.camera_width, height=args.camera_height)
                # cv2.imwrite("filter.png", event_image)
            except Empty:
                if last_events is None:
                    continue
                events = last_events
                timestamp = time.time()
            last_events = events
            start_time = time.time()
            if activity_filter:
                processed_events = activity_filter.process_events_gpu(events)
                if processed_events is None or processed_events["x"].size == 0:
                    continue
                x_gpu = processed_events['x']
                y_gpu = processed_events['y']
            else:
                x_gpu = cp.asarray(events['x'], dtype=cp.int32)
                y_gpu = cp.asarray(events['y'], dtype=cp.int32)
            if pixel_margin > 0:
                mask = (x_gpu >= pixel_margin) & (x_gpu < (frame_width - pixel_margin))
                x_filtered_gpu = x_gpu[mask]
                y_filtered_gpu = y_gpu[mask]
            else:
                x_filtered_gpu = x_gpu
                y_filtered_gpu = y_gpu
            current_time = time.time()
            valid_boxes = [box for box in boxes_shared if not np.allclose(box, [SENTINEL]*4)]
            count = len(valid_boxes)
            if valid_boxes:
                last_bbox = np.array(valid_boxes)
                last_bbox_time = current_time
                radar_viewer.update_max_objects(max(2,count))
            else:
                if last_bbox is not None and (current_time - last_bbox_time) < 1:
                    pass
                else:
                    boxes_shared[:] = SENTINEL
                    last_bbox = None
                    x_filtered_gpu = cp.asarray([], dtype=cp.int32)
                    y_filtered_gpu = cp.asarray([], dtype=cp.int32)
                    

            activity_monitor.process_events(x_filtered_gpu)
            activity_monitor.reset()
            ev_rate_gpu = activity_monitor.get_ev_rate_per_bin(return_to_cpu=False)
            total_events = activity_monitor.get_total_event_count()
            if total_events < config.low_event_threshold:
                no_movement_counter += 1
                if no_movement_counter > 5:
                    last_valid_angles = []
                    radar_viewer.angle_velocities = {}
            else:
                no_movement_counter = 0
            radar_img = update_radar_visualization(ev_rate_gpu)
            proc_time = round((time.time() - start_time) * 1000, 2)
            if radar_queue.full():
                try:
                    radar_queue.get_nowait()
                except Empty:
                    pass
            try:
                print(proc_time)
                radar_queue.put((radar_img, timestamp, proc_time))
            except Full:
                continue
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error in person direction process: {e}")
    finally:
        running.clear()

def fall_detection_process(frame_queue, fall_queue, running):
    try:
        fall_detector = PeopleFall()
        last_frame = None
        
        while running.is_set():
            try:
                event_frame, timestamp, _ = frame_queue.get(timeout=0.1)
                frame = event_frame.data
                h, w, _ = frame.shape
                if h != 320 or w != 320:
                    frame = cv2.resize(frame, (320, 320))
                    
                last_frame = frame
            except Empty:
                if last_frame is None:
                    continue
                frame = last_frame
                timestamp = time.time()

            start_time = time.time()
            # Run fall detection
            fall_status = fall_detector.process_frame(frame)
            end_time = time.time()
            processing_time = round((end_time - start_time) * 1000, 2)
            if fall_queue.full():
                try:
                    fall_queue.get_nowait()
                except Empty:
                    pass

            try:
                fall_queue.put((fall_status, timestamp, processing_time))
            except Full:
                continue

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error in fall detection process: {e}")
    finally:
        running.clear()
        
def visualization_process(people_queue, fall_queue, radar_queue, result_queue, running, frame_height, args: ProcessingParams):
    TARGET_HEIGHT = frame_height
    fall_display_counter = 0

    last_stats_update = time.time()
    current_fps = 0
    current_people_time = 0
    current_radar_time = 0 
    current_fall_time = 0
    update_interval = 1.0  
    
    try:
        while running.is_set():
            try:
                # Get data from queues
                frame, boxes, count, p_timestamp, people_time = people_queue.get(timeout=0.01)
                frame_radar, r_timestamp, radar_time = radar_queue.get(timeout=0.01)
                try:
                    fall_status, f_timestamp, fall_time = fall_queue.get_nowait()
                except Empty:
                    fall_status = "Normal"
                    fall_time = 0

                # 1. Process event image (left side)
                event_target_height = TARGET_HEIGHT
                event_target_width = TARGET_HEIGHT  # Keep it square as original
                
                event_image_resized = cv2.resize(frame, 
                                            (event_target_width, event_target_height),
                                            interpolation=cv2.INTER_AREA)
                
                # 2. Process radar image
                radar_target_height = int(TARGET_HEIGHT * 2/3)
                radar_aspect_ratio = frame_radar.shape[1] / frame_radar.shape[0]
                radar_target_width = int(radar_target_height * radar_aspect_ratio)
                
                if radar_target_width > event_target_width:
                    radar_target_width = event_target_width
                    radar_target_height = int(radar_target_width / radar_aspect_ratio)
                
                radar_image_resized = cv2.resize(frame_radar,
                    (radar_target_width, radar_target_height),
                    interpolation=cv2.INTER_AREA)
                
                # 3. Create right side canvas
                right_side = np.zeros((TARGET_HEIGHT, event_target_width, 3), dtype=np.uint8)
    
                # Draw radar on top
                x_offset = (event_target_width - radar_target_width) // 2
                y_offset = 0
                right_side[y_offset:y_offset+radar_target_height, 
                        x_offset:x_offset+radar_target_width] = radar_image_resized

                # Stats area
                stats_height = TARGET_HEIGHT - radar_target_height
                stats_y = radar_target_height
                
                # Split stats area into two parts
                stats_width = event_target_width
                stats_mid = stats_width // 2
                
                # Fill background
                right_side[stats_y:, :] = [255, 255, 255]
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_stats_update >= update_interval:
                    max_time = max(people_time, 0, fall_time)
                    if max_time > 0:
                        current_fps = 1000 / max_time
                    else:
                        current_fps = 0
                    
                    current_people_time = people_time
                    current_radar_time = 0
                    current_fall_time = fall_time
                    
                    last_stats_update = current_time

                # Setup font params based on TARGET_HEIGHT
                font = cv2.FONT_HERSHEY_SIMPLEX
                base_font_scale = TARGET_HEIGHT / 720.0  # Scale based on 720p
                text_color = (0, 0, 0)
                line_gap = int(40 * base_font_scale)
                margin = int(20 * base_font_scale)
                base_thickness = max(1, int(2 * base_font_scale))

                # Left side stats
                stats_x = margin
                text_y = stats_y + int(40 * base_font_scale)

                # Status with color
                
                print("count fall: ", fall_display_counter)
                if fall_status == "FALL":
                    fall_display_counter += 1

                if fall_display_counter > 3:
                    fall_text = "Status: Fall Detected!"
                    fall_color = (0, 0, 255)
                    cv2.putText(right_side, fall_text,
                              (margin, text_y), font,
                              0.8 * base_font_scale, fall_color, base_thickness + 1, cv2.LINE_AA)
                    fall_display_counter += 1
                    if fall_display_counter > 20:
                        fall_display_counter = 0
                else:
                    fall_text = "Status: Normal"
                    fall_color = (0, 255, 0)
                    cv2.putText(right_side, fall_text,
                              (margin, text_y), font,
                              0.8 * base_font_scale, fall_color, base_thickness + 1, cv2.LINE_AA)

                # People count
                text_y += line_gap
                cv2.putText(right_side, f"People: {count}",
                           (stats_x, text_y), font,
                           1.2 * base_font_scale, text_color, base_thickness + 1, cv2.LINE_AA)

                # FPS
                if hasattr(args, 'show_fps') and args.show_fps:
                    text_y += line_gap
                    cv2.putText(right_side, f"FPS: {current_fps:.1f}",
                              (stats_x, text_y), font,
                              0.8 * base_font_scale, text_color, base_thickness , cv2.LINE_AA)

                # Right side timing stats
                if hasattr(args, 'show_timing') and args.show_timing:
                    # Draw vertical divider
                    divider_thickness = max(1, int(2 * base_font_scale))
                    cv2.line(right_side,
                            (stats_mid, stats_y),
                            (stats_mid, TARGET_HEIGHT),
                            (200, 200, 200), divider_thickness)

                    timing_x = stats_mid + margin
                    text_y = stats_y + int(40 * base_font_scale)

                    # Title
                    cv2.putText(right_side, "Processing Time (ms):",
                              (timing_x, text_y), font,
                              0.7 * base_font_scale, text_color, base_thickness, cv2.LINE_AA)

                    # Timing details
                    text_y += line_gap
                    detail_scale = 0.7 * base_font_scale
                    cv2.putText(right_side, f"Detection: {current_people_time:.1f}",
                              (timing_x, text_y), font,
                              detail_scale, text_color, base_thickness, cv2.LINE_AA)

                    text_y += line_gap
                    cv2.putText(right_side, f"Radar: {current_radar_time:.1f}",
                              (timing_x, text_y), font,
                              detail_scale, text_color, base_thickness, cv2.LINE_AA)

                    text_y += line_gap
                    cv2.putText(right_side, f"Fall: {current_fall_time:.1f}",
                              (timing_x, text_y), font,
                              detail_scale, text_color, base_thickness, cv2.LINE_AA)

                # Combine images
                combined_image = np.hstack((event_image_resized, right_side))

                # Handle queue
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except Empty:
                        pass

                try:
                    result_queue.put((combined_image, frame, count, fall_status, time.time()))
                except Full:
                    continue

            except Empty:
                continue

    except Exception as e:
        logging.error(f"Error in visualization process: {e}")
    finally:
        running.clear()

def all_one_thread(result_queue, input_filename, camera_resolution, model_path, args, frame_height):
    try:
        bbox_shm = get_bbox_shm()
        boxes_shared = np.ndarray((MAX_BBOXES, 4), dtype=np.float32, buffer=bbox_shm.buf)
        detector = ONNXDetector(model_path, args)
        fall_detector = PeopleFall()
        TARGET_HEIGHT = frame_height
        fall_display_counter = 0
        fall_status = "Normal"
        
        last_valid_angles = []
        no_movement_counter = 0
        last_frame_time = time.time()
        current_bbox = []
        last_peak_time = time.time()
        frame_width = args.camera_width
        frame_height = args.camera_height
        radar_height = int(frame_height * 0.95)
        radar_width = int(frame_width * 1.47)
        
        config = RadarConfig()
        lateral_fov_deg = np.rad2deg(config.lateral_fov)
        pixel_margin = int(config.trim_angle_deg / (lateral_fov_deg / frame_width))
        activity_monitor = ActivityMonitor(config, frame_width)
        radar_viewer = RadarViewer(config, radar_width, radar_height)
        
        if config.use_filter:
            activity_filter = ActivityNoiseFilterGPU(
                width=frame_width,
                height=frame_height,
                threshold=config.delta_t,
                neighborhood_size=args.neighborhood_size,
                time_tolerance=args.time_tolerance
            )
        else:
            activity_filter = None
            
        def update_radar_visualization(ev_rate_gpu: cp.ndarray) -> np.ndarray:
            nonlocal last_frame_time, last_valid_angles, last_peak_time
            radar_img = np.copy(radar_viewer.linear_bins_cpu)
            draw_radar_background(radar_img, radar_width, radar_height, config.lateral_fov)
            frame_delta = time.time() - last_frame_time
            last_frame_time = time.perf_counter()
            if no_movement_counter > 10 and last_valid_angles:
                angles_to_show = [float(a + v) for a, v in zip(last_valid_angles, radar_viewer.angle_velocities)]
            else:
                angles_found = radar_viewer.compute_view(ev_rate_gpu, radar_img, frame_delta)
                angles_to_show = radar_viewer.stabilize_angles(angles_found)
                if angles_to_show:
                    last_valid_angles = angles_to_show.copy()
                    last_peak_time = time.time()  
            if (not angles_to_show or len(angles_to_show) == 0) and (time.time() - last_peak_time > 1):
                radar_viewer.tracked_peaks.clear()
                last_valid_angles = []
                draw_radar_background(radar_img, radar_width, radar_height, config.lateral_fov)
            
            return radar_img
        
        w, h = camera_resolution
        processor = EventProcessor(
            width=w,
            height=h,
            crop_coordinates=None
        )
        
        iterator = BiasEventsIterator(
            delta_t=33000,
            input_filename=input_filename,
            bias_file=os.path.join(config_path, 'bias.bias')
        )
        
        frame_count = 0
        total_time = 0
        for events in iterator:
            start_time = time.time()
            
            event_frame = processor.create_frame(events)
            frame = event_frame.data
            
            boxes, boxes_n, current_count = detector.detect_and_track(frame)
            
            tracked_frame = frame.copy()
            if current_count > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(tracked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(tracked_frame, "Person", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            events_filtered = convert_events_data(events.copy())
            
                
            if activity_filter:
                processed_events = activity_filter.process_events_gpu(events_filtered)
                if processed_events is None or processed_events["x"].size == 0:
                    x_gpu = cp.array([], dtype=cp.int32)
                    y_gpu = cp.array([], dtype=cp.int32)
                else:
                    x_gpu = processed_events['x']
                    y_gpu = processed_events['y']
            else:
                x_gpu = cp.asarray(events_filtered['x'], dtype=cp.int32)
                y_gpu = cp.asarray(events_filtered['y'], dtype=cp.int32)
                
            if pixel_margin > 0 and x_gpu.size > 0:
                mask = (x_gpu >= pixel_margin) & (x_gpu < (frame_width - pixel_margin))
                x_filtered_gpu = x_gpu[mask]
                y_filtered_gpu = y_gpu[mask]
            else:
                x_filtered_gpu = x_gpu
                y_filtered_gpu = y_gpu
                
            current_time = time.time()
            valid_boxes = [box for box in boxes_shared if not np.allclose(box, [SENTINEL]*4)]
            count = len(valid_boxes)
            if valid_boxes:
                last_bbox = np.array(valid_boxes)
                last_bbox_time = current_time
                radar_viewer.update_max_objects(max(2,count))
            else:
                if last_bbox is not None and (current_time - last_bbox_time) < 1:
                    pass
                else:
                    boxes_shared[:] = SENTINEL
                    last_bbox = None
                    x_filtered_gpu = cp.asarray([], dtype=cp.int32)
                    y_filtered_gpu = cp.asarray([], dtype=cp.int32)

                    
            activity_monitor.process_events(x_filtered_gpu)
            activity_monitor.reset()
            ev_rate_gpu = activity_monitor.get_ev_rate_per_bin(return_to_cpu=False)
            total_events = activity_monitor.get_total_event_count()
            
            if total_events < config.low_event_threshold:
                no_movement_counter += 1
                if no_movement_counter > 5:
                    last_valid_angles = []
                    radar_viewer.angle_velocities = {}
            else:
                no_movement_counter = 0
                
            radar_img = update_radar_visualization(ev_rate_gpu)
            
            frame_fall = frame.copy()
            h, w, _ = frame_fall.shape
            if h != 320 or w != 320:
                frame_fall = cv2.resize(frame_fall, (320, 320))
            
            fall_status = fall_detector.process_frame(frame_fall)
            
            event_target_height = TARGET_HEIGHT
            event_target_width = TARGET_HEIGHT
            
            event_image_resized = cv2.resize(tracked_frame, 
                                        (event_target_width, event_target_height),
                                        interpolation=cv2.INTER_AREA)
            
            radar_target_height = int(TARGET_HEIGHT * 2/3)
            radar_aspect_ratio = radar_img.shape[1] / radar_img.shape[0]
            radar_target_width = int(radar_target_height * radar_aspect_ratio)
            
            if radar_target_width > event_target_width:
                radar_target_width = event_target_width
                radar_target_height = int(radar_target_width / radar_aspect_ratio)
            
            radar_image_resized = cv2.resize(radar_img,
                (radar_target_width, radar_target_height),
                interpolation=cv2.INTER_AREA)
            
            right_side = np.zeros((TARGET_HEIGHT, event_target_width, 3), dtype=np.uint8)
            
            x_offset = (event_target_width - radar_target_width) // 2
            y_offset = 0
            right_side[y_offset:y_offset+radar_target_height, 
                    x_offset:x_offset+radar_target_width] = radar_image_resized
            
            stats_height = TARGET_HEIGHT - radar_target_height
            stats_y = radar_target_height
            stats_width = event_target_width
            stats_mid = stats_width // 2
            
            right_side[stats_y:, :] = [255, 255, 255]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            base_font_scale = TARGET_HEIGHT / 720.0
            text_color = (0, 0, 0)
            line_gap = int(40 * base_font_scale)
            margin = int(20 * base_font_scale)
            base_thickness = max(1, int(2 * base_font_scale))
            
            stats_x = margin
            text_y = stats_y + int(40 * base_font_scale)
            
            print("count fall: ", fall_display_counter)
            
            if fall_status == "FALL":
                fall_display_counter += 1
                
            if fall_display_counter > 3:
                fall_text = "Status: Fall Detected!"
                fall_color = (0, 0, 255)
                cv2.putText(right_side, fall_text,
                            (margin, text_y), font,
                            0.8 * base_font_scale, fall_color, base_thickness + 1, cv2.LINE_AA)
                fall_display_counter += 1
                if fall_display_counter > 20:
                    fall_display_counter = 0
            else:
                fall_text = "Status: Normal"
                fall_color = (0, 255, 0)
                cv2.putText(right_side, fall_text,
                            (margin, text_y), font,
                            0.8 * base_font_scale, fall_color, base_thickness + 1, cv2.LINE_AA)
                            
            text_y += line_gap
            cv2.putText(right_side, f"People: {current_count}",
                        (stats_x, text_y), font,
                        1.2 * base_font_scale, text_color, base_thickness + 1, cv2.LINE_AA)
            
            combined_image = np.hstack((event_image_resized, right_side))
            
            end_time = time.time()
            processing_time = round((end_time - start_time) * 1000, 2)
            
            frame_count += 1
            total_time += processing_time
            avg_time = total_time / frame_count
            
            if hasattr(args, 'show_fps') and args.show_fps:
                fps = 1000 / avg_time if avg_time > 0 else 0
                text_y += line_gap
                cv2.putText(combined_image, f"FPS: {fps:.1f}",
                          (stats_x, text_y), font,
                          0.8 * base_font_scale, text_color, base_thickness, cv2.LINE_AA)
                
                if frame_count % 100 == 0:
                    print(f"Frame: {frame_count}, Avg Processing Time: {avg_time:.1f}ms, FPS: {fps:.1f}")
            
            while True:
                try:
                    result_queue.put((combined_image, frame, count, fall_status, time.time()), block=True, timeout=0.1)
                    break
                except Full:
                    time.sleep(0.01) 
        
        print(f"File processing completed. Processed {frame_count} frames")
        print(f"Average processing time: {total_time/frame_count:.2f}ms")
        
        result_queue.put(("EOF", None, 0, "Normal", time.time()))
        
    except Exception as e:
        print(traceback.print_exc())
        logging.error(f"Error in event process: {e}")
    finally:
        if 'iterator' in locals() and iterator:
            del iterator
               
class StreamProcessor:
    def __init__(self, frame_queue_size=5, resize_dims=(320, 320), target_fps=30, input_filename="", args: ProcessingParams = None):
        # Queues for inter-process communication
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
        self.model_path = os.path.join(yolo_path, "best_s_22_1.onnx")
        
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
            args=(self.result_queue_single, self.input_filename, self.camera_resolution, self.model_path, self.args, self.frame_height),
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
            args=(self.frame_queue, self.frame_queue_fall, self.event_queue, self.running, self.input_filename, self.camera_resolution),
            daemon=True
        )
        
        self.people_process = Process(
            name="PeopleProcess",
            target=people_detection_process_onnx, 
            args=(self.frame_queue, self.people_queue, self.running, self.model_path, self.args),
            daemon=True
        )
        
        self.fall_process = Process(
            name="FallProcess",
            target=fall_detection_process,
            args=(self.frame_queue, self.fall_queue, self.running),
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

    def start(self):
        self.running.set()
        
        # Start all processes
        self.event_process = Process(
            name="EventProcess",
            target=event_process, 
            args=(self.frame_queue, self.event_queue, self.running, self.input_filename, self.camera_resolution),
            daemon=True
        )
        
        self.people_process = Process(
            name="PeopleProcess",
            target=people_detection_process, 
            args=(self.frame_queue, self.people_queue, self.running, self.model_path, self.args),
            daemon=True
        )
        
        self.fall_process = Process(
            name="FallProcess",
            target=fall_detection_process,
            args=(self.frame_queue, self.fall_queue, self.running),
            daemon=True
        )
        
        self.person_direction = Process(
            name="PersonDirection",
            target=person_direction_process,
            args=(self.event_queue, self.radar_queue, self.running, self.args),
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
        
        last_result = None
        
        try:
            while self.running.is_set():
                try:
                    result = self.result_queue.get(timeout=0.1)
                    last_result = result
                except Empty:
                    if last_result is None:
                        continue
                    result = last_result

                tracked_frame, _, current_count, fall_status, _ = result

                stable_count = self.update_people_count(current_count)
                
                # Draw stats and display
                # tracked_frame = self.draw_stats(tracked_frame, fps, stable_count, fall_status)
                cv2.imshow("WQ EVS TIVI", tracked_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

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

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}")
    try:
        current_process = multiprocessing.current_process()
        for child in current_process.children():
            child.terminate()
        
        for child in current_process.children():
            child.join(timeout=3)
    except Exception as e:
        print(f"Error in signal handler: {e}")
    
    sys.exit(0)
    
def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    processor = None
    try:
        params = ProcessingParams()
        processor = StreamProcessor(args=params)
        processor.start()
    except Exception as e:
        traceback.print_exc()
        print(f"Error occurred: {e}")
    finally:
        if processor:
            processor.stop()

def cleanup_old_processes():
    import psutil
    current_process = psutil.Process()
    process_name = current_process.name()
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == process_name and proc.pid != current_process.pid:
                psutil.Process(proc.pid).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
        
    restart_usb_devices()
    cleanup_old_processes()
    main()
