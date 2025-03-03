# radar.py
import cupy as cp
import numpy as np
import cv2
from cupyx.scipy import signal
from scipy.optimize import linear_sum_assignment
import time
from src.domain.value_objects.config import RadarConfig
from src.infrastructure.utils.utils import interpolate_bin

class ActivityMonitor:
    def __init__(self, config: RadarConfig, sensor_width: int):
        self.conf = config
        self.n_bins = config.n_bins
        self.bin_width = sensor_width / self.n_bins
        self.histogram = cp.zeros(self.n_bins, dtype=cp.float32)
        self.prev_histogram = None

    def process_events(self, x_gpu: cp.ndarray):
        if x_gpu.size == 0:
            return
        x_bins = cp.floor(x_gpu / self.bin_width).astype(cp.int32)
        x_bins = cp.clip(x_bins, 0, self.n_bins - 1)
        counts = cp.bincount(x_bins, minlength=self.n_bins).astype(cp.float32)
        self.histogram += counts

    def reset(self):
        if self.histogram is not None:
            self.prev_histogram = cp.copy(self.histogram)
        self.histogram.fill(0)

    def get_ev_rate_per_bin(self, return_to_cpu=True):
        if self.prev_histogram is None:
            zeros = cp.zeros(self.n_bins, dtype=cp.float32)
            return zeros.get() if return_to_cpu else zeros
        ev_rate_per_bin_gpu = self.prev_histogram / (self.conf.delta_t * 1e-6)
        return ev_rate_per_bin_gpu.get() if return_to_cpu else ev_rate_per_bin_gpu

    def get_total_event_count(self) -> float:
        if self.prev_histogram is None:
            return 0.0
        total_gpu = cp.sum(self.prev_histogram)
        return float(total_gpu.get())

class RadarViewer:
    def __init__(self, config, width, height, max_missed_frames=0, max_objects = 3):
        self.conf = config
        self.max_objects = max_objects
        self.width = width
        self.height = height
        self.n_bins = config.n_bins
        self.n_bins_x = config.n_bins_x
        self.n_bins_y = config.n_bins_y
        self.lateral_fov = config.lateral_fov
        self.min_ev_rate = config.min_ev_rate
        self.max_ev_rate = config.max_ev_rate
        self.max_range = config.max_range
        self.min_peak_prominence = config.min_peak_prominence
        self.last_update_time = 0
        self.update_interval = 0.01
        self.target_peaks = []
        self.peak_move_threshold = 30.0
        self.max_missed_frames = max_missed_frames
        self.tracked_peaks = {}
        self.radar_center_x = self.width // 2
        self.radar_center_y = self.height
        self.scale_factor = self.height / self.conf.max_range
        self.mapx, self.mapy = self.compute_radar_maps_gpu()
        self.linear_bins_cpu = np.zeros((height, width, 3), dtype=np.uint8)
        self.reset_grid()
        self.stabilized_angles = cp.array([], dtype=cp.float32)
        self.radar_bg = np.zeros((height, width, 3), dtype=np.uint8)
        self.last_angles = []
        self.angle_velocities = {}
        self.max_dist_change = 20
        self.next_peak_id = 0
        self.k_spring = 3
        self.c_damp = 1
        self.peak_velocities = {}
        self.max_angle_velocity = np.deg2rad(20.0)
        self.max_dist_velocity = 20.0
        self.angle_cost_weight = 2
        self.dist_cost_weight = 1.5
        self.predefined_max_threshold = 150.0
    def update_max_objects(self, new_max_objects: int):
        self.max_objects = new_max_objects
        
    def wrap_angle(self, angle_rad):
        half_fov = self.lateral_fov / 2
        angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
        return np.clip(angle_rad, -half_fov, half_fov)

    def constrain_distance(self, distance):
        return np.clip(distance, 0, self.max_range)

    def match_peaks_hungarian(self, new_peaks, current_peaks):
        if not new_peaks or not current_peaks:
            return [], list(range(len(new_peaks))), list(range(len(current_peaks)))
        
        try:
            new_angles = cp.array([peak[0] for peak in new_peaks], dtype=cp.float32)
            new_dists = cp.array([peak[1] for peak in new_peaks], dtype=cp.float32)
            curr_angles = cp.array([peak[0] for peak in current_peaks], dtype=cp.float32)
            curr_dists = cp.array([peak[1] for peak in current_peaks], dtype=cp.float32)
            
            angle_diff_matrix = cp.abs(new_angles[:, None] - curr_angles[None, :])
            
            half_fov = self.lateral_fov / 2
            angle_diff_matrix = (angle_diff_matrix + cp.pi) % (2 * cp.pi) - cp.pi
            angle_diff_matrix = cp.clip(angle_diff_matrix, -half_fov, half_fov)
            
            angle_diff_deg = cp.rad2deg(angle_diff_matrix)
            dist_diff_matrix = cp.abs(new_dists[:, None] - curr_dists[None, :])
            
            angle_cost = self.angle_cost_weight * angle_diff_deg
            dist_cost = self.dist_cost_weight * dist_diff_matrix
            
            over_angle = angle_diff_deg > self.peak_move_threshold
            if cp.any(over_angle):
                angle_cost[over_angle] += self.angle_cost_weight * (
                    angle_diff_deg[over_angle] - self.peak_move_threshold
                )
            
            over_dist = dist_diff_matrix > self.max_dist_change
            if cp.any(over_dist):
                dist_cost[over_dist] += self.dist_cost_weight * (
                    dist_diff_matrix[over_dist] - self.max_dist_change
                )
            
            total_cost = angle_cost + dist_cost
            
            finite_mask = total_cost < 1e6
            if cp.any(finite_mask):
                finite_costs = total_cost[finite_mask]
                mean_cost = float(cp.mean(finite_costs).get())
                std_cost = float(cp.std(finite_costs).get())
                dynamic_threshold = min(mean_cost + 2 * std_cost, self.predefined_max_threshold)
                self.conf.mot_gating_threshold = dynamic_threshold
            else:
                self.conf.mot_gating_threshold = self.predefined_max_threshold
            
            cost_matrix = cp.full((len(new_peaks), len(current_peaks)), 1e6, dtype=cp.float32)
            mask = total_cost < self.conf.mot_gating_threshold
            cost_matrix[mask] = total_cost[mask]
            
            cost_matrix_cpu = cost_matrix.get()
            
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu)
            except ValueError as e:
                return [], list(range(len(new_peaks))), list(range(len(current_peaks)))
            
            matched_pairs = []
            unmatched_new = set(range(len(new_peaks)))
            unmatched_current = set(range(len(current_peaks)))
            
            for r, c in zip(row_ind, col_ind):
                if r < len(new_peaks) and c < len(current_peaks) and cost_matrix_cpu[r, c] < 1e6:
                    matched_pairs.append((r, c))
                    unmatched_new.discard(r)
                    unmatched_current.discard(c)
            
            result_peaks = []
            for new_idx, curr_idx in matched_pairs:
                new_angle, new_dist = new_peaks[new_idx]
                curr_id = current_peaks[curr_idx][2]
                result_peaks.append((new_angle, new_dist, curr_id))
            
            return result_peaks, list(unmatched_new), list(unmatched_current)
            
        except Exception as e:
            return [], list(range(len(new_peaks))), list(range(len(current_peaks)))


    def compute_radar_maps_gpu(self):
        mapx = cp.full((self.height, self.width), -1, dtype=cp.int32)
        mapy = cp.full((self.height, self.width), -1, dtype=cp.int32)
        j_indices, i_indices = cp.meshgrid(cp.arange(self.width), cp.arange(self.height))
        x_c = self.width / 2.0
        y_c = self.height
        theta = (j_indices / self.width - 0.5) * self.lateral_fov
        r = self.height - i_indices
        x_pol = cp.round(x_c + r * cp.sin(theta)).astype(cp.int32)
        y_pol = cp.round(y_c - r * cp.cos(theta)).astype(cp.int32)
        valid_mask = (x_pol >= 0) & (x_pol < self.width) & (y_pol >= 0) & (y_pol < self.height)
        mapx[y_pol[valid_mask], x_pol[valid_mask]] = j_indices[valid_mask]
        mapy[y_pol[valid_mask], x_pol[valid_mask]] = i_indices[valid_mask]
        return mapx, mapy

    def reset_grid(self):
        self.linear_bins_cpu.fill(0)
        bin_width = self.width // self.n_bins_x
        bin_height = self.height // self.n_bins_y
        for i in range(self.n_bins_x + 1):
            x = i * bin_width
            cv2.line(self.linear_bins_cpu, (x, 0), (x, self.height), (0, 255, 0), 1)
        for k in range(self.n_bins_y + 1):
            y = k * bin_height
            cv2.line(self.linear_bins_cpu, (0, y), (self.width, y), (0, 255, 0), 1)

    def compute_real_angle(self, computed_angle_rad):
        real_angle_rad = computed_angle_rad + self.lateral_fov / 2
        real_angle_deg = np.rad2deg(real_angle_rad) + 45
        return real_angle_deg

    def stabilize_angles(self, angles):
        if len(angles) != len(self.stabilized_angles):
            self.stabilized_angles = cp.asarray(angles)
            return angles
        angles_gpu = cp.asarray(angles)
        stable = []
        for i, a in enumerate(angles_gpu):
            if i < len(self.stabilized_angles):
                prev = self.stabilized_angles[i]
                new_a = self.conf.stability_smooth_alpha * a + (1 - self.conf.stability_smooth_alpha) * prev
            else:
                new_a = a
            stable.append(float(new_a))
        self.stabilized_angles = cp.asarray(stable)
        return stable

    def update_peak_positions(self, delta_t):
        if not self.tracked_peaks:
            return
        try:
            peak_ids = list(self.tracked_peaks.keys())
            positions = cp.array([
                (data['angle'], data['distance']) 
                for data in self.tracked_peaks.values()
            ], dtype=cp.float32)
            
            velocities = cp.array([
                self.peak_velocities.get(pid, (0.0, 0.0)) 
                for pid in peak_ids
            ], dtype=cp.float32)
            curr_angles = positions[:, 0]
            curr_dists = positions[:, 1]
            v_angles = velocities[:, 0]
            v_dists = velocities[:, 1]
            half_fov = self.lateral_fov / 2
            curr_angles = (curr_angles + cp.pi) % (2 * cp.pi) - cp.pi
            curr_angles = cp.clip(curr_angles, -half_fov, half_fov)
            angle_diffs = cp.zeros_like(curr_angles)
            dist_diffs = cp.zeros_like(curr_dists)
            
            force_angles = self.k_spring * angle_diffs
            force_dists = self.k_spring * dist_diffs

            angle_damps = self.c_damp * (1 + 0.5 * cp.abs(v_angles) / self.max_angle_velocity)
            dist_damps = self.c_damp * (1 + 0.5 * cp.abs(v_dists) / self.max_dist_velocity)
            
            v_angles_new = v_angles + (force_angles - angle_damps * v_angles) * delta_t
            v_dists_new = v_dists + (force_dists - dist_damps * v_dists) * delta_t
            
            v_angles_new = cp.clip(
                v_angles_new, 
                -self.max_angle_velocity, 
                self.max_angle_velocity
            )
            v_dists_new = cp.clip(
                v_dists_new,
                -self.max_dist_velocity,
                self.max_dist_velocity
            )
            
            angles_new = curr_angles + v_angles_new * delta_t
            dists_new = curr_dists + v_dists_new * delta_t
            
            angles_new = (angles_new + cp.pi) % (2 * cp.pi) - cp.pi
            angles_new = cp.clip(angles_new, -half_fov, half_fov)
            dists_new = cp.clip(dists_new, 0, self.max_range)
            
            max_angle_change = cp.deg2rad(15.0)
            angle_changes = cp.abs(angles_new - curr_angles)
            large_changes = angle_changes > max_angle_change
            if cp.any(large_changes):
                angles_new[large_changes] = curr_angles[large_changes] + \
                    cp.sign(angles_new[large_changes] - curr_angles[large_changes]) * max_angle_change
                v_angles_new[large_changes] *= 0.5
            
            positions_new = cp.stack([angles_new, dists_new], axis=1)
            velocities_new = cp.stack([v_angles_new, v_dists_new], axis=1)
            
            positions_cpu = positions_new.get()
            velocities_cpu = velocities_new.get()
            
            updated_peaks = {}
            for i, pid in enumerate(peak_ids):
                updated_peaks[pid] = {
                    'angle': float(positions_cpu[i, 0]),
                    'distance': float(positions_cpu[i, 1]),
                    'missed': self.tracked_peaks[pid]['missed']
                }
                self.peak_velocities[pid] = (
                    float(velocities_cpu[i, 0]),
                    float(velocities_cpu[i, 1])
                )
            
            self.tracked_peaks = updated_peaks
            
        except Exception as e:
            pass

    def match_and_update_peaks(self, new_peaks, delta_t):
        current_peaks_list = [(data['angle'], data['distance'], pid) 
                                for pid, data in self.tracked_peaks.items()]
        matched_peaks, unmatched_new, unmatched_current = self.match_peaks_hungarian(new_peaks, current_peaks_list)
        updated_peaks = {}
        for angle, distance, pid in matched_peaks:
            updated_peaks[pid] = {
                'angle': angle,
                'distance': distance,
                'missed': 0
            }
        merge_threshold = 0.1  
        for i in sorted(unmatched_new, key=lambda i: new_peaks[i][1], reverse=True):
            new_angle, new_distance = new_peaks[i]
            duplicate_found = False
            for peak in updated_peaks.values():
                if abs(peak['angle'] - new_angle) < merge_threshold:
                    duplicate_found = True
                    break
            if not duplicate_found and len(updated_peaks) < self.max_objects:
                updated_peaks[self.next_peak_id] = {
                    'angle': new_angle,
                    'distance': new_distance,
                    'missed': 0
                }
                self.next_peak_id += 1
        for j in unmatched_current:
            existing_pid = list(self.tracked_peaks.keys())[j]
            if existing_pid in self.tracked_peaks:
                new_missed = self.tracked_peaks[existing_pid]['missed'] + 1
                if new_missed <= self.max_missed_frames:
                    updated_peaks[existing_pid] = {
                        'angle': self.tracked_peaks[existing_pid]['angle'],
                        'distance': self.tracked_peaks[existing_pid]['distance'],
                        'missed': new_missed
                    }
        self.tracked_peaks = updated_peaks
        self.update_peak_positions(delta_t)

    def compute_view(self, histo_gpu, radar_image_cpu, delta_t):
        current_time = time.time()
        for peak_id, peak_data in self.tracked_peaks.items():
            self.draw_peak_on_radar(radar_image_cpu, peak_data['angle'], peak_data['distance'], peak_id)
        if current_time - self.last_update_time < self.update_interval:
            return self.last_angles
        self.last_update_time = current_time
        new_peaks = []
        new_angles = []
        if histo_gpu.size > 0:
            peaks_gpu, _ = signal.find_peaks(histo_gpu, prominence=self.min_peak_prominence)
            if len(histo_gpu) > 1:
                if (histo_gpu[0] > (histo_gpu[1] + self.min_peak_prominence)) and (0 not in peaks_gpu):
                    peaks_gpu = cp.append(peaks_gpu, 0)
                if (histo_gpu[-1] > (histo_gpu[-2] + self.min_peak_prominence)) and ((len(histo_gpu)-1) not in peaks_gpu):
                    peaks_gpu = cp.append(peaks_gpu, len(histo_gpu) - 1)
            if len(peaks_gpu) > 0:
                peak_values_gpu = histo_gpu[peaks_gpu]
                valid_mask = (peak_values_gpu >= self.min_ev_rate) & (peak_values_gpu <= self.max_ev_rate)
                valid_peaks_gpu = peaks_gpu[valid_mask]
                valid_values_gpu = peak_values_gpu[valid_mask]
                if len(valid_peaks_gpu) > 0:
                    sorted_idx = cp.argsort(valid_values_gpu)[::-1]
                    if len(sorted_idx) > self.max_objects:
                        sorted_idx = sorted_idx[:self.max_objects]
                    position_idx = cp.argsort(valid_peaks_gpu[sorted_idx])
                    final_peaks_gpu = valid_peaks_gpu[sorted_idx][position_idx]
                    final_values_gpu = valid_values_gpu[sorted_idx][position_idx]
                    final_peaks_cpu = final_peaks_gpu.get()
                    final_values_cpu = final_values_gpu.get()
                    histo_cpu = histo_gpu.get()
                    for peak, val in zip(final_peaks_cpu, final_values_cpu):
                        peak_idx = int(peak)
                        bin_position = interpolate_bin(histo_cpu, peak_idx)
                        angle_rad = (bin_position / (self.n_bins - 1) - 0.5) * self.lateral_fov
                        real_angle_deg = self.compute_real_angle(angle_rad)
                        new_angles.append(real_angle_deg)
                        intensity_ratio = (val - self.min_ev_rate) / (self.max_ev_rate - self.min_ev_rate)
                        intensity_ratio = np.clip(intensity_ratio, 0, 1)
                        distance = self.conf.calculate_distance(intensity_ratio, return_cpu=True)
                        new_peaks.append((angle_rad, distance))
        if len(new_peaks) > 0:
            self.match_and_update_peaks(new_peaks, delta_t)
        smoothed = self.stabilize_angles(new_angles)
        self.last_angles = smoothed
        return smoothed

    def draw_peak_on_radar(self, radar_image_cpu, angle_rad, distance, peak_id):
        angle_rad = self.wrap_angle(angle_rad)
        distance = self.constrain_distance(distance)
        scaled_distance = distance * self.scale_factor
        x = int(self.radar_center_x + scaled_distance * np.sin(angle_rad))
        y = int(self.radar_center_y - scaled_distance * np.cos(angle_rad))
        color = (0, 0, 255)
        cv2.circle(radar_image_cpu, (x, y), 8, color, -1)
        cv2.circle(radar_image_cpu, (x, y), 6, color, -1)
        cv2.circle(radar_image_cpu, (x, y), 4, color, -1)
        cv2.circle(radar_image_cpu, (x, y), 2, color, -1)
        display_angle = np.rad2deg(angle_rad) + 45
        # if int(distance)>95:
        #     distance = f">1m"
        # else:
        #     distance = f"{distance:.1f}cm"
        # # text = f"{display_angle:.1f}o\n{distance}"
        text = f"{display_angle:.1f}o"
        y_offset = y
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(radar_image_cpu, text, (x + 10, y_offset+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        y_offset += 15