# src/infrastructure/setup/component_setup.py

import logging
from typing import Tuple, Any, Optional
from src.application.services.config_service import SystemConfig
from src.infrastructure.event_camera.iterators.bias_events_iterator import BiasEventsIterator
from src.infrastructure.event_camera.iterators.video_frame_iterator import VideoFrameIterator
from src.infrastructure.event_camera.processors.event_processor import EventProcessor
from src.infrastructure.ml_models.yolo_detector import YoloDetector
from src.infrastructure.tracking.byte_tracker import ByteTracker
from src.infrastructure.counting.line_counter import LineCounter
from src.infrastructure.counting.frame_counter import FrameCounter
from src.infrastructure.visualization.display import DisplayProcessor
from src.application.services.video_detection_service import VideoDetectionService
from src.application.services.detection_service import DetectionService

logger = logging.getLogger(__name__)

def setup_input_components(config: SystemConfig) -> Tuple[Any, Optional[EventProcessor]]:
    """Initialize input components based on input type"""
    if config.input.type == "video":
        iterator = VideoFrameIterator(
            input_source=config.input.file_path,
            target_fps=config.input.target_fps if hasattr(config.input, 'target_fps') else None,
            resize_dims=(config.input.width, config.input.height)
        )
        processor = None
        logger.info("Initialized video frame iterator")
    else:  # event camera
        iterator = BiasEventsIterator(
            delta_t=10000,
            input_filename=config.input.file_path if config.input.file_path else None,
            bias_file=config.input.bias_file if hasattr(config.input, 'bias_file') else None
        )
        processor = EventProcessor(
            width=config.input.width,
            height=config.input.height,
            crop_coordinates=None
        )
        logger.info("Initialized event iterator and processor")
    
    return iterator, processor

def setup_detector(config: SystemConfig) -> YoloDetector:
    """Initialize YOLO detector"""
    detector = YoloDetector(
        model_path=config.model.path,
        conf_threshold=config.model.confidence_threshold
    )
    logger.info(f"Loaded detection model from {config.model.path}")
    return detector

def setup_tracker(config: SystemConfig) -> ByteTracker:
    """Initialize object tracker"""
    tracker = ByteTracker(
        max_age=config.tracking.max_age,
        min_hits=config.tracking.min_hits,
        iou_threshold=config.tracking.iou_threshold
    )
    logger.info("Initialized tracker")
    return tracker

def setup_counter(config: SystemConfig) -> Any:
    """Initialize appropriate counter based on configuration"""
    if config.counting.type == "line":
        counter = LineCounter()
        logger.info("Initialized line counter")
    else:
        counter = FrameCounter(
            alpha=config.counting.alpha,
            count_threshold=config.counting.count_threshold,
            temporal_window=config.counting.temporal_window
        )
        logger.info("Initialized frame counter")
    return counter

def setup_display(config: SystemConfig) -> DisplayProcessor:
    """Initialize display processor"""
    display = DisplayProcessor(
        track_history_length=config.visualization.track_history_length,
        display_fps=config.visualization.display_fps
    )
    logger.info("Initialized display")
    return display

def setup_detection_service(
    config: SystemConfig,
    iterator: Any,
    processor: Optional[EventProcessor],
    detector: YoloDetector,
    tracker: ByteTracker,
    counter: Any
) -> Any:
    """Initialize appropriate detection service based on input type"""
    if config.input.type == "video":
        service = VideoDetectionService(
            frame_iterator=iterator,
            detector=detector,
            tracker=tracker,
            counter=counter
        )
        logger.info("Initialized video detection service")
    else:
        service = DetectionService(
            event_iterator=iterator,
            detector=detector,
            tracker=tracker,
            counter=counter,
            event_processor=processor
        )
        logger.info("Initialized event detection service")
    
    return service

def setup_all_components(config: SystemConfig) -> Tuple[Any, Any, YoloDetector, ByteTracker, Any, DisplayProcessor]:
    """Initialize all system components based on configuration"""
    try:
        # Initialize input components
        iterator, processor = setup_input_components(config)
        
        # Initialize other components
        detector = setup_detector(config)
        tracker = setup_tracker(config)
        counter = setup_counter(config)
        display = setup_display(config)
        
        return iterator, processor, detector, tracker, counter, display

    except Exception as e:
        logger.error(f"Error during component initialization: {str(e)}")
        raise