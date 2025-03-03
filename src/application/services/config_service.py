import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path

from src.domain.entities.config import CountingConfig, CountingZoneConfig, InputConfig, ModelConfig, SystemConfig, TrackingConfig, VisualizationConfig


class ConfigService:
    @staticmethod
    def load_config(config_path: str = "configs/default_config.yaml") -> SystemConfig:
        """Load system configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert dictionary to typed config objects
        input_config = InputConfig(
            type=config_dict['input'].get('type', 'event'),
            file_path=config_dict['input']['file_path'],
            width=config_dict['input']['width'],
            height=config_dict['input']['height'],
            target_fps=config_dict['input'].get('target_fps'),
            bias_file=config_dict['input'].get('bias_file'),
            delta_t=config_dict['input'].get('delta_t', 10000),
            crop_coordinates=config_dict['input'].get('crop_coordinates')
        )
        
        model_config = ModelConfig(**config_dict['model'])
        tracking_config = TrackingConfig(**config_dict['tracking'])
        visualization_config = VisualizationConfig(**config_dict['visualization'])
        
        # Convert zone configs
        zones = [
            CountingZoneConfig(**zone)
            for zone in config_dict['counting']['zones']
        ]
        
        # Create counting config
        counting_config = CountingConfig(
            type=config_dict['counting']['type'],
            zones=zones,
            alpha=config_dict['counting'].get('alpha', 0.3),
            count_threshold=config_dict['counting'].get('count_threshold', 2),
            temporal_window=config_dict['counting'].get('temporal_window', 5)
        )

        return SystemConfig(
            input=input_config,
            model=model_config,
            tracking=tracking_config,
            visualization=visualization_config,
            counting=counting_config
        )

    @staticmethod
    def get_default_config() -> SystemConfig:
        """Get default configuration when no config file is provided"""
        return ConfigService.load_config()

    @staticmethod
    def validate_config(config: SystemConfig) -> bool:
        """Validate configuration values"""
        try:
            # Validate input type
            if config.input.type not in ["video", "event"]:
                raise ValueError(f"Invalid input type: {config.input.type}")

            # Validate input file path
            if not Path(config.input.file_path).exists():
                raise FileNotFoundError(f"Input file not found: {config.input.file_path}")

            # Validate dimensions
            if config.input.width <= 0 or config.input.height <= 0:
                raise ValueError("Invalid dimensions")

            # Validate bias file if provided
            if config.input.bias_file and not Path(config.input.bias_file).exists():
                raise FileNotFoundError(f"Bias file not found: {config.input.bias_file}")

            # Validate model path
            if not Path(config.model.path).exists():
                raise FileNotFoundError(f"Model file not found: {config.model.path}")

            # Validate confidence threshold
            if not 0 <= config.model.confidence_threshold <= 1:
                raise ValueError("Confidence threshold must be between 0 and 1")

            # Validate tracking parameters
            if config.tracking.max_age <= 0:
                raise ValueError("max_age must be positive")
            if config.tracking.min_hits <= 0:
                raise ValueError("min_hits must be positive")
            if not 0 <= config.tracking.iou_threshold <= 1:
                raise ValueError("iou_threshold must be between 0 and 1")

            # Validate counting parameters
            if config.counting.type not in ["line", "frame"]:
                raise ValueError(f"Invalid counting type: {config.counting.type}")

            if config.counting.type == "line" and not config.counting.zones:
                raise ValueError("Line counting requires at least one zone")

            return True

        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False