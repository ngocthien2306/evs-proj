# utils.py
import cupy as cp
import numpy as np
import cv2
import time
from typing import Tuple

from src.core.filters import BoundingBox

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


def resize_with_padding(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    return padded_image

def pad_frames(event_frame: np.ndarray, radar_frame: np.ndarray, target_height: int) -> Tuple[np.ndarray, np.ndarray]:
    def pad_frame(frame: np.ndarray, target_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if h < target_h:
            pad_bottom = target_h - h
            return cv2.copyMakeBorder(frame, 0, pad_bottom, 0, 0,
                                      cv2.BORDER_CONSTANT, value=(0,0,0))
        return frame
    padded_event = pad_frame(event_frame, target_height)
    padded_radar = pad_frame(radar_frame, target_height)
    return padded_event, padded_radar

def filter_noise_gpu(events_gpu: cp.ndarray, threshold: float = 0.1) -> cp.ndarray:
    if events_gpu.size < 2:
        return events_gpu
    temporal_mask = cp.concatenate(([False], events_gpu['t'][1:] - events_gpu['t'][:-1] > threshold))
    spatial_density = cp.histogramdd(
        cp.column_stack((events_gpu['x'], events_gpu['y'])),
        bins=[32, 32]
    )[0]
    density_mask = spatial_density[events_gpu['y'], events_gpu['x']] > cp.mean(spatial_density)
    return events_gpu[temporal_mask & density_mask]

def process_events_batch(events: np.ndarray, batch_size: int = 1000000) -> np.ndarray:
    total_events = len(events)
    processed_events = []
    for i in range(0, total_events, batch_size):
        batch = events[i:min(i + batch_size, total_events)]
        batch_gpu = cp.asarray(batch)
        filtered_batch = filter_noise_gpu(batch_gpu)
        processed_events.extend(cp.asnumpy(filtered_batch))
    return np.array(processed_events, dtype=events.dtype)

class FPSCounter:
    def __init__(self, avg_window: int = 30):
        self.avg_window = avg_window
        self.frame_times = []
        self.last_time = time.time()
        self.fps = 0.0

    def update(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > self.avg_window:
            self.frame_times.pop(0)
        total_time = sum(self.frame_times)
        self.fps = len(self.frame_times) / total_time if total_time > 0 else 0.0

    def get_fps(self) -> float:
        return self.fps

def draw_radar_background(radar_image_cpu: np.ndarray, width: int, height: int, lateral_fov: float) -> None:
    radar_image_cpu.fill(0)
    center = (width // 2, height)
    max_radius = height
    num_circles = 5
    angles = np.linspace(-lateral_fov / 2, lateral_fov / 2, num=7)
    for angle_rad in angles:
        end_x = int(center[0] + max_radius * np.sin(angle_rad))
        end_y = int(center[1] - max_radius * np.cos(angle_rad))
        cv2.line(radar_image_cpu, center, (end_x, end_y), 
                (0, 255, 0), 1, cv2.LINE_AA)
        if angle_rad != 0:
            angle_deg = int(np.rad2deg(angle_rad + lateral_fov / 2) + 0.5)
            label_x = int(center[0] + (max_radius - 30) * np.sin(angle_rad))
            label_y = int(center[1] - (max_radius - 30) * np.cos(angle_rad))
            cv2.putText(radar_image_cpu, f"{angle_deg}",
                      (label_x, label_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for i in range(num_circles):
        radius = int(max_radius * (i + 1) / num_circles)
        for j in range(len(angles) - 1):
            start_angle = angles[j]
            end_angle = angles[j + 1]
            arc_angles = np.linspace(start_angle, end_angle, num=50)
            points = np.array([[int(center[0] + radius * np.sin(a)), 
                              int(center[1] - radius * np.cos(a))] 
                             for a in arc_angles], dtype=np.int32)
            if len(points) > 1:
                for k in range(len(points) - 1):
                    cv2.line(radar_image_cpu, 
                            tuple(points[k]), 
                            tuple(points[k + 1]), 
                            (0, 255, 0), 1, cv2.LINE_AA)
        # distance = int((i + 1) * 100 / num_circles)
        # cv2.putText(radar_image_cpu, f"{distance}cm", 
        #            (center[0] + 10, center[1] - radius),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def interpolate_bin(histo_cpu: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(histo_cpu) -1:
        return float(peak_idx)
    left = histo_cpu[peak_idx -1]
    center = histo_cpu[peak_idx]
    right = histo_cpu[peak_idx +1]
    denominator = (left - 2 * center + right)
    if denominator == 0:
        return float(peak_idx)
    delta = 0.5 * (left - right) / denominator
    return float(peak_idx) + delta
def make_binary_histo(events, img=None, width=304, height=240):
    """
    simple display function that shows negative events as blacks dots and positive as white one
    on a gray background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        img[events['y'], events['x'], :] = 255 * events['p'][:, None]
    return img


def process_events_batch(events, activity_filter, use_filter):
    if use_filter and activity_filter is not None:
        processed_events = activity_filter.process_events_gpu(events)
        if processed_events is None or processed_events.size == 0:
            return None
        return processed_events
    return events

def process_gpu_operations(x_gpu, y_gpu, frame_width, frame_height, pixel_margin):
    combined_mask = (
        (x_gpu >= pixel_margin) &
        (x_gpu < frame_width - pixel_margin) &
        (y_gpu < frame_height)
    )
    return x_gpu[combined_mask], y_gpu[combined_mask]

def create_event_image_gpu(x_gpu, y_gpu, frame_height, frame_width):
    event_image_gpu = cp.zeros((frame_height, frame_width, 3), dtype=cp.uint8)
    valid_indices = (y_gpu < frame_height) & (x_gpu < frame_width)
    y_valid = y_gpu[valid_indices]
    x_valid = x_gpu[valid_indices]
    if y_valid.size > 0:
        event_image_gpu[y_valid, x_valid] = 255
    return event_image_gpu

def resize_with_padding(image, target_size):
    """
    Resize the image to target size while keeping the aspect ratio and padding with black.
    """
    h, w = image.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image

def resize_image(image, target_size):
    """
    Resize the image to target size while keeping the aspect ratio.
    No padding is applied.
    """
    h, w = image.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image

def process_visualization(event_image_bgr, radar_image_cpu, frame_width, combined_height, target_height):
    """
    Create a composite image with configurable output height
    """
    start_time = time.time()
    
    # Use configurable target height
    TARGET_HEIGHT = target_height
    
    # Record timing for each operation
    timings = {}
    
    # 1. Process event image (left side)
    t_start = time.time()
    event_target_height = TARGET_HEIGHT
    event_target_width = TARGET_HEIGHT  # Keep it square as original
    
    event_image_resized = cv2.resize(event_image_bgr, 
                                   (event_target_width, event_target_height),
                                   interpolation=cv2.INTER_AREA)
    timings['event_resize'] = (time.time() - t_start) * 1000
    
    # 2. Process radar image
    t_start = time.time()
    radar_target_height = int(TARGET_HEIGHT * 2/3)
    radar_aspect_ratio = radar_image_cpu.shape[1] / radar_image_cpu.shape[0]
    radar_target_width = int(radar_target_height * radar_aspect_ratio)
    
    if radar_target_width > event_target_width:
        radar_target_width = event_target_width
        radar_target_height = int(radar_target_width / radar_aspect_ratio)
    
    radar_image_resized = cv2.resize(radar_image_cpu,
                                   (radar_target_width, radar_target_height),
                                   interpolation=cv2.INTER_AREA)
    timings['radar_resize'] = (time.time() - t_start) * 1000
    
    # 3. Create right side canvas
    t_start = time.time()
    right_side = np.zeros((TARGET_HEIGHT, event_target_width, 3), dtype=np.uint8)
    
    x_offset = (event_target_width - radar_target_width) // 2
    y_offset = 0
    
    right_side[y_offset:y_offset+radar_target_height, 
              x_offset:x_offset+radar_target_width] = radar_image_resized
    
    right_side[radar_target_height:, :] = [255, 255, 255]
    timings['compose_right'] = (time.time() - t_start) * 1000
    
    # 4. Combine images
    t_start = time.time()
    combined_image = np.hstack((event_image_resized, right_side))
    timings['combine'] = (time.time() - t_start) * 1000
    
    total_time = time.time() - start_time
    timings['total'] = total_time * 1000
    
    return combined_image, timings

def convert_events_dtype(events):
    converted_events = np.empty(len(events), dtype=[
        ('x', '<u2'),
        ('y', '<u2'),
        ('p', 'u1'),
        ('t', '<u4')
    ])
    if 't' in events.dtype.names:
        converted_events['t'] = events['t']
    else:
        converted_events['t'] = events['timestamp']
    converted_events['x'] = events['x']
    converted_events['y'] = events['y']
    if 'p' in events.dtype.names:
        converted_events['p'] = events['p']
    else:
        converted_events['p'] = events['polarity']
    
    return converted_events
