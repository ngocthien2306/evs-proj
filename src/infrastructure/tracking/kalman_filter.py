from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.domain.entities.detection import BoundingBox, Detection
from src.application.interfaces.tracker import ITracker

class KalmanFilter:
    """Kalman Filter for bounding box tracking"""
    def __init__(self):
        # State: [x, y, w, h, dx, dy, dw, dh]
        self.state_dim = 8
        self.meas_dim = 4  # Measurement: [x, y, w, h]

        # State transition matrix
        self.F = np.eye(self.state_dim)
        self.F[:4, 4:] = np.eye(4)  # Add velocity components

        # Measurement matrix
        self.H = np.zeros((self.meas_dim, self.state_dim))
        self.H[:4, :4] = np.eye(4)

        # Process noise
        self.Q = np.eye(self.state_dim) * 0.1
        self.Q[4:, 4:] *= 4.0  # Larger noise for velocity components

        # Measurement noise
        self.R = np.eye(self.meas_dim) * 1.0

        # Error covariance
        self.P = np.eye(self.state_dim) * 10.0

        self.x = None

    def initialize(self, measurement: np.ndarray) -> None:
        """Initialize state with first measurement"""
        self.x = np.zeros(self.state_dim)
        self.x[:4] = measurement
        self.x[4:] = 0  # Initialize velocities to 0

    def predict(self) -> np.ndarray:
        """Predict next state"""
        if self.x is None:
            return None
            
        # Predict state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:4]  # Return predicted measurement

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with measurement"""
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        y = measurement - (self.H @ self.x)
        self.x = self.x + (K @ y)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

        return self.x[:4]