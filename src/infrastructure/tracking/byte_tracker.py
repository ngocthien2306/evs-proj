from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.domain.entities.detection import BoundingBox, Detection
from src.application.interfaces.tracker import ITracker
from .track import Track

class ByteTracker(ITracker):
    """ByteTrack implementation for multi-object tracking"""
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        track_thresh: float = 0.5,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_thresh = track_thresh
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.tracks: List[Track] = []
        self.track_id = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracks with new detections"""
        # Split detections by confidence
        high_conf_dets = [d for d in detections if d.bbox.confidence >= self.track_high_thresh]
        low_conf_dets = [d for d in detections if self.track_low_thresh <= d.bbox.confidence < self.track_high_thresh]

        # Get predictions from existing tracks
        track_boxes = []
        for track in self.tracks:
            track_boxes.append(track.predict())

        # Initialize unmatched tracks list
        unmatched_tracks_a = list(range(len(track_boxes)))  # Add this line

        # First association with high confidence detections
        if track_boxes and high_conf_dets:
            matches_a, unmatched_tracks_a, unmatched_dets_high = \
                self._match_detections_to_tracks(high_conf_dets, track_boxes)

            # Update matched tracks
            for track_idx, det_idx in matches_a:
                self.tracks[track_idx].update(high_conf_dets[det_idx])

            # Handle unmatched tracks
            for track_idx in unmatched_tracks_a:
                self.tracks[track_idx].time_since_update += 1
        else:
            unmatched_dets_high = list(range(len(high_conf_dets)))
            for track in self.tracks:
                track.time_since_update += 1

        # Second association with low confidence detections
        if unmatched_tracks_a and low_conf_dets:  # This will now always be properly defined
            unmatched_track_boxes = [track_boxes[i] for i in unmatched_tracks_a]
            matches_b, unmatched_tracks_b, unmatched_dets_low = \
                self._match_detections_to_tracks(low_conf_dets, unmatched_track_boxes)

            # Update matched tracks
            for local_track_idx, det_idx in matches_b:
                global_track_idx = unmatched_tracks_a[local_track_idx]
                self.tracks[global_track_idx].update(low_conf_dets[det_idx])

        # Initialize new tracks
        if high_conf_dets:  # Add this check
            for det_idx in unmatched_dets_high:
                self._initiate_track(high_conf_dets[det_idx])

        # Remove dead tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update < self.max_age
        ]

        # Return current tracked objects
        results = []
        for track in self.tracks:
            if track.time_since_update > 1 or track.hits < self.min_hits:
                continue

            bbox = track.predict()
            if bbox is None:
                continue

            results.append(Detection(
                bbox=bbox,
                track_id=track.track_id,
                keypoints=track.detection.keypoints
            ))

        return results

    def _match_detections_to_tracks(
        self,
        detections: List[Detection],
        track_boxes: List[BoundingBox]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using IoU and Hungarian algorithm"""
        if not track_boxes or not detections:
            return [], list(range(len(track_boxes))), list(range(len(detections)))

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(track_boxes), len(detections)))
        for t, track_box in enumerate(track_boxes):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track_box, detection.bbox)

        # Use Hungarian algorithm for optimal assignment
        track_indices, detection_indices = linear_sum_assignment(-iou_matrix)

        matches = []
        unmatched_tracks = list(range(len(track_boxes)))
        unmatched_detections = list(range(len(detections)))

        # Filter matches by IoU threshold
        for t, d in zip(track_indices, detection_indices):
            if iou_matrix[t, d] >= self.iou_threshold:
                matches.append((t, d))
                if t in unmatched_tracks:
                    unmatched_tracks.remove(t)
                if d in unmatched_detections:
                    unmatched_detections.remove(d)

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection: Detection) -> None:
        """Create and initialize a new track"""
        if detection.bbox.confidence >= self.track_thresh:
            self.tracks.append(Track(detection, self.track_id))
            self.track_id += 1

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0