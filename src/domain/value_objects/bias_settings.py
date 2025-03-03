# src/domain/value_objects/bias_settings.py
from dataclasses import dataclass
from typing import Dict, Optional
import os
from ..entities.camera_bias import CameraBias

@dataclass
class BiasSettings:
    DEFAULT_BIASES = {
        "bias_diff": (300, (300, 300)),
        "bias_diff_off": (225, (0, 299)),
        "bias_diff_on": (375, (301, 1800)),
        "bias_fo": (1725, (1650, 1800)),
        "bias_hpf": (1500, (0, 1800)),
        "bias_pr": (1500, (1200, 1800)),
        "bias_refr": (1500, (1300, 1700)),
    }

    biases: Dict[str, CameraBias]
    current_bias_name: str

    @classmethod
    def create_default(cls) -> 'BiasSettings':
        biases = {
            name: CameraBias(name, value, limits)
            for name, (value, limits) in cls.DEFAULT_BIASES.items()
        }
        return cls(biases=biases, current_bias_name=next(iter(biases)))

    @classmethod
    def from_dict(cls, bias_dict: Dict[str, int]) -> 'BiasSettings':
        biases = {}
        for name, value in bias_dict.items():
            if name in cls.DEFAULT_BIASES:
                limits = cls.DEFAULT_BIASES[name][1]
                biases[name] = CameraBias(name, value, limits)
        return cls(biases=biases, current_bias_name=next(iter(biases)))

    @classmethod
    def from_file(cls, file_path: str) -> 'BiasSettings':
        """Load bias settings from a file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Bias file not found: {file_path}")
            
        bias_dict = {}
        with open(file_path, "r") as file:
            for line in file.readlines():
                line = line.split("%")
                if len(line[0].strip()):
                    bias_dict[line[1].strip()] = int(line[0].strip())
        
        return cls.from_dict(bias_dict)

    def cycle_current_bias(self) -> str:
        current_idx = list(self.biases.keys()).index(self.current_bias_name)
        next_idx = (current_idx + 1) % len(self.biases)
        self.current_bias_name = list(self.biases.keys())[next_idx]
        return self.current_bias_name