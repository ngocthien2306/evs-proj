# filters.py
import cupy as cp
import numpy as np
from typing import List

class BoundingBox:
    def __init__(self, x_center: float, y_center: float, width: float, height: float, 
                 frame_width: int, frame_height: int, expansion_ratio: float = -0.2):
        self.x_center = x_center * frame_width
        self.y_center = y_center * frame_height
        self.width = width * frame_width
        self.height = height * frame_height
        self.x_min = int(self.x_center - self.width / 2)
        self.x_max = int(self.x_center + self.width / 2)
        self.y_min = int(self.y_center - self.height / 2)
        self.y_max = int(self.y_center + self.height / 2)
        self.expand(expansion_ratio, frame_width, frame_height)
    
    def expand(self, ratio: float, frame_width: int, frame_height: int):
        expand_width = self.width * abs(ratio)
        expand_height = self.height * abs(ratio)
        if ratio < 0:
            self.x_min = int(min(max(self.x_min + expand_width / 2, 0), frame_width - 1))
            self.x_max = int(max(min(self.x_max - expand_width / 2, frame_width - 1), 0))
            self.y_min = int(min(max(self.y_min + expand_height / 2, 0), frame_height - 1))
            self.y_max = int(max(min(self.y_max - expand_height / 2, frame_height - 1), 0))
        else:
            self.x_min = int(max(self.x_min - expand_width / 2, 0))
            self.x_max = int(min(self.x_max + expand_width / 2, frame_width - 1))
            self.y_min = int(max(self.y_min - expand_height / 2, 0))
            self.y_max = int(min(self.y_max + expand_height / 2, frame_height - 1))
    
    def contains(self, x: int, y: int) -> bool:
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)
# class BoundingBox:
#     def __init__(self, x_center: float, y_center: float, width: float, height: float, 
#                  frame_width: int, frame_height: int, expansion_ratio: float = -0.2,
#                  expansion_ratio_x: float = None, expansion_ratio_y: float = None):
#         self.x_center = x_center * frame_width
#         self.y_center = y_center * frame_height
#         self.width = width * frame_width
#         self.height = height * frame_height
#         self.x_min = int(self.x_center - self.width / 2)
#         self.x_max = int(self.x_center + self.width / 2)
#         self.y_min = int(self.y_center - self.height / 2)
#         self.y_max = int(self.y_center + self.height / 2)
#         # Nếu không có giá trị riêng cho x hay y thì dùng expansion_ratio chung
#         if expansion_ratio_x is None:
#             expansion_ratio_x = expansion_ratio
#         if expansion_ratio_y is None:
#             expansion_ratio_y = expansion_ratio
#         self.expand(expansion_ratio_x, expansion_ratio_y, frame_width, frame_height)
    
#     def expand(self, expansion_ratio_x: float, expansion_ratio_y: float, frame_width: int, frame_height: int):
#         # Xử lý mở rộng theo chiều x
#         expand_width = self.width * abs(expansion_ratio_x)
#         if expansion_ratio_x < 0:
#             self.x_min = int(min(max(self.x_min + expand_width / 2, 0), frame_width - 1))
#             self.x_max = int(max(min(self.x_max - expand_width / 2, frame_width - 1), 0))
#         else:
#             self.x_min = int(max(self.x_min - expand_width / 2, 0))
#             self.x_max = int(min(self.x_max + expand_width / 2, frame_width - 1))
        
#         # Xử lý mở rộng theo chiều y
#         expand_height = self.height * abs(expansion_ratio_y)
#         if expansion_ratio_y < 0:
#             self.y_min = int(min(max(self.y_min + expand_height / 2, 0), frame_height - 1))
#             self.y_max = int(max(min(self.y_max - expand_height / 2, frame_height - 1), 0))
#         else:
#             self.y_min = int(max(self.y_min - expand_height / 2, 0))
#             self.y_max = int(min(self.y_max + expand_height / 2, frame_height - 1))
    
#     def contains(self, x: int, y: int) -> bool:
#         return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

def filter_events(x: np.ndarray, y: np.ndarray, bounding_boxes: List[BoundingBox], max_threads: int = 3):
    if x.size == 0 or len(bounding_boxes) == 0:
        return np.array([], dtype=x.dtype), np.array([], dtype=y.dtype)
    
    def filter_single_box(bb: BoundingBox):
        mask = ((x >= bb.x_min) & (x <= bb.x_max) &
                (y >= bb.y_min) & (y <= bb.y_max))
        return mask
    
    combined_mask = filter_single_box(bounding_boxes[0])
    for bb in bounding_boxes[1:max_threads]:
        mask = filter_single_box(bb)
        combined_mask = np.logical_or(combined_mask, mask)
    
    x_filtered = x[combined_mask]
    y_filtered = y[combined_mask]
    return x_filtered, y_filtered

class ActivityNoiseFilterGPU:
    def __init__(self, width: int, height: int, threshold: int, neighborhood_size: int = 5, time_tolerance: int = 1000):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.neighborhood_size = neighborhood_size
        self.time_tolerance = time_tolerance
        padding = neighborhood_size // 2
        self.w_plus_pad = width + 2 * padding
        self.h_plus_pad = height + 2 * padding
        self.last_ts_gpu = cp.full((self.w_plus_pad * self.h_plus_pad,), 0, dtype=cp.uint32)
        self.activity_filter_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void activityFilterKernel(
            const unsigned int* t_arr,
            const int* x_arr,
            const int* y_arr,
            unsigned int* last_ts, 
            bool* out_valid,
            const int num_events,
            const int w_plus_pad,
            const unsigned int threshold,
            const int neighborhood_size,
            const unsigned int time_tolerance
        ) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= num_events) return;
            unsigned int t_i = t_arr[i];
            int x_i = x_arr[i];
            int y_i = y_arr[i];
            int center_idx = (y_i + neighborhood_size/2) * w_plus_pad + (x_i + neighborhood_size/2);
            unsigned int time_min = (t_i > threshold + time_tolerance) ? (t_i - threshold - time_tolerance) : 0;
            unsigned int time_max = t_i + time_tolerance;
            bool is_valid = false;
            int radius = neighborhood_size / 2;
            for (int dy = -radius; dy <= radius && !is_valid; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    unsigned int nb_ts = last_ts[center_idx + dy * w_plus_pad + dx];
                    if (nb_ts >= time_min && nb_ts <= time_max) {
                        is_valid = true;
                        break;
                    }
                }
            }
            last_ts[center_idx] = t_i;
            out_valid[i] = is_valid;
        }
        ''', 'activityFilterKernel')

    def process_events_gpu(self, events: np.ndarray):
        if events.size == 0:
            return None 

        event_fields = events.dtype.names

        t_gpu = cp.asarray(events['t'], dtype=cp.uint32)
        x_gpu = cp.asarray(events['x'], dtype=cp.int32)
        y_gpu = cp.asarray(events['y'], dtype=cp.int32)
        
        num_events = x_gpu.size
        out_valid = cp.zeros(num_events, dtype=cp.bool_)
        block_size = 256
        grid_size = (num_events + block_size - 1) // block_size
        
        self.activity_filter_kernel(
            (grid_size,), (block_size,),
            (t_gpu, x_gpu, y_gpu,
            self.last_ts_gpu,
            out_valid,
            num_events,
            self.w_plus_pad,
            self.threshold,
            self.neighborhood_size,
            self.time_tolerance)
        )
        
        valid_idx = cp.where(out_valid)[0]
        
        if 'p' in event_fields:
            p_gpu = cp.asarray(events['p'], dtype=cp.int16)  # Sử dụng int16 để giữ giá trị âm nếu có
            valid_p_mask = p_gpu[valid_idx] >= 0
            valid_idx = valid_idx[valid_p_mask]
        
        num_valid = valid_idx.size
        if num_valid == 0:
            return None 
        
        t_filtered_gpu = t_gpu[valid_idx]
        x_filtered_gpu = x_gpu[valid_idx]
        y_filtered_gpu = y_gpu[valid_idx]
        
        result = {'t': t_filtered_gpu, 'x': x_filtered_gpu, 'y': y_filtered_gpu}
        if 'p' in event_fields:
            result['p'] = p_gpu[valid_idx]
        return result

