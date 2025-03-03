from typing import Optional, Iterator, Any
import os.path
import metavision_hal as mv_hal
from metavision_core.event_io.raw_reader import RawReaderBase

from ....application.interfaces.event_iterator import IEventIterator
from ....domain.value_objects.bias_settings import BiasSettings
from ..exceptions.camera_exceptions import CameraNotFoundError, InvalidFileError

class NonBufferedEventIterator(IEventIterator):
    def __init__(self, delta_t: float, input_filename: Optional[str] = None,
                 bias_settings: Optional[BiasSettings] = None):
        self.__is_live = not input_filename

        if not self.__is_live:
            if not os.path.isfile(input_filename):
                raise InvalidFileError(f"Invalid input file: {input_filename}")
            self.reader = RawReaderBase(input_filename, delta_t=delta_t)
        else:
            device = mv_hal.DeviceDiscovery.open("")
            if not device:
                raise CameraNotFoundError("No live camera found")

            geometry = device.get_i_geometry()
            print(f"Camera resolution: {geometry.get_width()} x {geometry.get_height()}")

            if bias_settings:
                for name, bias in bias_settings.biases.items():
                    device.get_i_ll_biases().set(name, bias.value)

            self.reader = RawReaderBase("", device=device, delta_t=delta_t, 
                                      initiate_device=False)

    def __iter__(self) -> Iterator[Any]:
        while not self.reader.is_done():
            yield self.reader.load_delta_t(-1)

    def get_size(self) -> int:
        return self.reader.get_size()
