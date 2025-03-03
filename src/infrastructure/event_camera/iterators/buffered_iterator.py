from typing import Optional, Iterator, Any
import os.path
import metavision_hal as mv_hal
from metavision_core.event_io import EventsIterator

from ....application.interfaces.event_iterator import IEventIterator
from ....domain.value_objects.bias_settings import BiasSettings
from ..exceptions.camera_exceptions import CameraNotFoundError, InvalidFileError

class BufferedEventIterator(IEventIterator):
    def __init__(self, delta_t: float, input_filename: Optional[str] = None, 
                 bias_settings: Optional[BiasSettings] = None):
        self.__is_live = not input_filename

        if not self.__is_live:
            if not os.path.isfile(input_filename):
                raise InvalidFileError(f"Invalid input file: {input_filename}")
            self.__ev_it = EventsIterator(input_filename, delta_t=delta_t)
        else:
            device = mv_hal.DeviceDiscovery.open("")
            if not device:
                raise CameraNotFoundError("No live camera found")

            if bias_settings:
                for name, bias in bias_settings.biases.items():
                    device.get_i_ll_biases().set(name, bias.value)

            self.__ev_it = EventsIterator(device, delta_t=delta_t)

    def __iter__(self) -> Iterator[Any]:
        yield from self.__ev_it

    def get_size(self) -> int:
        return self.__ev_it.get_size()