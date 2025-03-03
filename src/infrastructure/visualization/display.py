import cv2
import numpy as np
from typing import List, Optional, Dict, Union
from src.domain.value_objects.counting_result import CountingResult
from src.domain.entities.detection import Frame
from src.domain.value_objects.counting_zone import CountingZone

class DisplayProcessor:
    def __init__(self,
                 window_name: str = "EVS Detection",
                 track_history_length: int = 30,
                 display_fps: bool = True):
        self.window_name = window_name
        self.colors = np.random.randint(0, 255, size=(100, 3))
        self.track_paths: Dict[int, List] = {}
        self.track_history_length = track_history_length
        self.display_fps = display_fps

    def display_frame(self,
                     frame: Frame,
                     counting_data: Optional[Union[List[CountingZone], CountingResult]] = None,
                     display_tracks: bool = True) -> bool:
        """
        Display frame with detections and counting information.
        Args:
            frame: Frame to display
            counting_data: Either List[CountingZone] for line counting or CountingResult for frame counting
            display_tracks: Whether to display tracking paths
        """
        display_img = frame.data.copy()

        if frame.detections:
            self._draw_detections(display_img, frame.detections, display_tracks)

        # Draw counting information based on type
        if counting_data is not None:
            if isinstance(counting_data, list):  # CountingZone list
                self._draw_counting_zones(display_img, counting_data)
            else:  # CountingResult
                self._draw_counting_result(display_img, counting_data)

        # Draw FPS if enabled
        if self.display_fps and frame.fps > 0:
            cv2.putText(
                display_img,
                f"FPS: {frame.fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(self.window_name, display_img)
        return not (cv2.waitKey(1) & 0xFF == ord('q'))
    
    def _draw_detections(self, image: np.ndarray, detections: List, display_tracks: bool) -> None:
        """
        Draw detections, tracking IDs, and track paths on the image.
        For non-tracked detections, just draw the bounding box.
        """
        for det in detections:
            if det.track_id is not None:
                color = self.colors[det.track_id % len(self.colors)].tolist()
            else:
                color = (0, 255, 0) 
                
            # Draw bounding box
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw confidence score
            conf_text = f"{det.bbox.confidence:.2f}"
            cv2.putText(
                image,
                "Person",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            if det.track_id is not None:
                id_text = f"ID: {det.track_id}"
                cv2.putText(
                    image,
                    id_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                # Handle track path if display_tracks is enabled
                if display_tracks:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Initialize track path if needed
                    if det.track_id not in self.track_paths:
                        self.track_paths[det.track_id] = []
                    
                    # Update track path
                    self.track_paths[det.track_id].append(center)
                    
                    # Limit path length using track_history_length
                    if len(self.track_paths[det.track_id]) > self.track_history_length:
                        self.track_paths[det.track_id].pop(0)

                    # Draw track path
                    points = np.array(self.track_paths[det.track_id])
                    if len(points) > 1:  # Only draw if we have at least 2 points
                        for i in range(1, len(points)):
                            # Make line thinner towards older points
                            thickness = int(np.sqrt(float(i + 1) * 2))
                            cv2.line(
                                image,
                                tuple(points[i-1]),
                                tuple(points[i]),
                                color,
                                thickness
                            )

            if display_tracks:
                current_tracks = {det.track_id for det in detections if det.track_id is not None}
                self.track_paths = {
                    k: v for k, v in self.track_paths.items() 
                    if k in current_tracks
                }
                
    def _draw_counting_zones(self, image: np.ndarray, zones: List[CountingZone]) -> None:
        for zone in zones:
            # Draw zone polygon
            points = np.array(zone.points, np.int32)
            cv2.polylines(image, [points], True, (0, 255, 255), 2)

            # Draw counts
            text = f"{zone.name} - In: {zone.in_count}, Out: {zone.out_count}"
            text_pos = (zone.points[0][0], zone.points[0][1] - 10)
            cv2.putText(
                image,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    def _draw_counting_result(self, image: np.ndarray, result: CountingResult) -> None:
        # Draw current count
        cv2.putText(
            image,
            f"Current Count: {result.current_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            1
        )
        
        # Draw smoothed count if available
        # if hasattr(result, 'smoothed_count'):
        #     cv2.putText(
        #         image,
        #         f"Smoothed Count: {result.smoothed_count}",
        #         (20, 120),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 255, 255),
        #         2
        #     )

        # Draw max count
        # cv2.putText(
        #     image,
        #     f"Max Count: {result.max_count}",
        #     (20, 160),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 255, 255),
        #     2
        # )