from dataclasses import dataclass

@dataclass
class ProcessingParams:
    def __init__(
        self,
        neighborhood_size: int = 3,
        time_tolerance: int = 1000,
        buffer_size: int = 256,
        use_filter: bool = True,
        output_height: int = 720,
        show_fps: bool = False,
        show_timing: bool = False,
        yolo_conf: float = 0.6,
        yolo_iou: float = 0.5,
        camera_width: int = 320,
        camera_height: int = 320
    ):
        self.neighborhood_size = neighborhood_size
        self.time_tolerance = time_tolerance
        self.buffer_size = buffer_size
        self.use_filter = use_filter
        self.output_height = output_height
        self.show_fps = show_fps
        self.show_timing = show_timing
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.camera_width = camera_width
        self.camera_height = camera_height

    def __str__(self):
        return (
            f"ProcessingParams("
            f"neighborhood_size={self.neighborhood_size}, "
            f"time_tolerance={self.time_tolerance}, "
            f"buffer_size={self.buffer_size}, "
            f"use_filter={self.use_filter}, "
            f"output_height={self.output_height}, "
            f"show_fps={self.show_fps}, "
            f"show_timing={self.show_timing}, "
            f"yolo_conf={self.yolo_conf}, "
            f"yolo_iou={self.yolo_iou}, "
            f"camera_width={self.camera_width}, "
            f"camera_height={self.camera_height})"
        )