# config.py

import cupy as cp
import numpy as np

class RadarConfig:
    """
    Configuration class for radar visualization and event processing.
    Manages parameters for event filtering, radar display, and object tracking.
    """
    def __init__(self):
        ## Old config
        self.n_bins = 15
        self.n_bins_x = 15
        # self.n_bins = 10
        # self.n_bins_x = 10
        self.n_bins_y = 5
        self.lateral_fov = np.pi / 2  
        
        self.min_ev_rate_unfiltered = 5e3
        self.max_ev_rate_unfiltered = 1e7
        self.min_peak_prominence_unfiltered = 1e4
        self.low_event_threshold_unfiltered = 5e3
        
        ## New config
        # self.min_ev_rate_filtered = 1e3
        # self.max_ev_rate_filtered = 1e7
        # self.min_peak_prominence_filtered = 6e3
        # self.low_event_threshold_filtered = 4e3
        # Old config (better)
        self.min_ev_rate_filtered = 8e2
        self.max_ev_rate_filtered = 1e7
        self.min_peak_prominence_filtered = 5e3
        self.low_event_threshold_filtered = 1e3
        
        self.use_filter = True
        self.min_ev_rate = self.min_ev_rate_filtered
        self.max_ev_rate = self.max_ev_rate_filtered
        self.min_peak_prominence = self.min_peak_prominence_filtered
        self.low_event_threshold = self.low_event_threshold_filtered
        
        self.delta_t = 33000 
        self.max_range = 100.0  
        self.trim_angle_deg = 1.0  
        self.max_objects = 3  
        
        self.stability_smooth_alpha = 1 
        self.frame_history = 10  
        self.angle_history = cp.zeros((self.frame_history,), dtype=cp.float32)
        self.angle_diff_threshold = 15.0 
        
        self.mot_gating_threshold = 40.0  
    
    def set_filter_mode(self, use_filter: bool) -> None:
        """
        Updates event processing thresholds based on filter mode.
        
        Args:
            use_filter (bool): If True, uses filtered mode thresholds; otherwise, uses unfiltered thresholds.
        """
        if use_filter:
            self.min_ev_rate = self.min_ev_rate_filtered
            self.max_ev_rate = self.max_ev_rate_filtered
            self.min_peak_prominence = self.min_peak_prominence_filtered
            self.low_event_threshold = self.low_event_threshold_filtered
        else:
            self.use_filter = False
            self.min_ev_rate = self.min_ev_rate_unfiltered
            self.max_ev_rate = self.max_ev_rate_unfiltered
            self.min_peak_prominence = self.min_peak_prominence_unfiltered
            self.low_event_threshold = self.low_event_threshold_unfiltered
    
    def calculate_distance(self, intensity_ratio: float, return_cpu: bool = True) -> float:
        """
        Calculates distance based on event intensity ratio.
        
        Args:
            intensity_ratio (float): Ratio of event intensity (0-1).
            return_cpu (bool): If True, returns a CPU float; otherwise, returns a GPU array.
        
        Returns:
            float or cupy.ndarray: Calculated distance in centimeters.
        """
        distance = (1.0 - intensity_ratio) * self.max_range
        return float(distance.get()) if return_cpu and isinstance(distance, cp.ndarray) else distance
