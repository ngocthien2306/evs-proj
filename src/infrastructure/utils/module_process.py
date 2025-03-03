import logging
import traceback
import cv2
import time
import numpy as np
from queue import Full, Empty

from src.infrastructure.utils.system_utils import restart_usb_devices
from src.infrastructure.ml_models.fall_detector import PeopleFall
from src.infrastructure.ml_models.onnx_detect import ONNXDetector
from src.domain.value_objects.config import RadarConfig
from src.domain.entities.entities import ProcessingParams
from src.core.filters import ActivityNoiseFilterGPU
from src.core.radar import ActivityMonitor, RadarViewer
from src.infrastructure.event_camera.iterators.bias_events_iterator import BiasEventsIterator
from src.infrastructure.event_camera.processors.event_processor import EventProcessor
from src.infrastructure.utils.utils import convert_events_data, draw_radar_background, make_binary_histo
import cupy as cp
import os
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
     
def event_process(frame_queue, frame_queue_frame, event_queue , running, input_filename, camera_resolution, config_path):
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
            bias_file=config_path
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
                            bias_file=config_path
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

def fall_detection_process(frame_queue, fall_queue, running, fall_model_path):
    try:
        fall_detector = PeopleFall(fall_model_path)
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

def all_one_thread(result_queue, input_filename, camera_resolution, model_path, args, frame_height, config_path, fall_model_path):
    try:
        bbox_shm = get_bbox_shm()
        boxes_shared = np.ndarray((MAX_BBOXES, 4), dtype=np.float32, buffer=bbox_shm.buf)
        detector = ONNXDetector(model_path, args)
        fall_detector = PeopleFall(fall_model_path)
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
            bias_file=config_path
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
   