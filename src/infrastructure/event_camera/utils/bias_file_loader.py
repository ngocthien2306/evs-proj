from typing import Dict
from ....domain.value_objects.bias_settings import BiasSettings

def load_bias_file(path: str) -> BiasSettings:
    """Load bias values from text file and return BiasSettings"""
    bias_dict = {}
    with open(path, "r") as file:
        for line in file.readlines():
            line = line.split("%")
            if len(line[0].strip()):
                bias_dict[line[1].strip()] = int(line[0].strip())
    return BiasSettings.from_dict(bias_dict)
