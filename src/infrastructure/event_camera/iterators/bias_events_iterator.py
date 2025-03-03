# src/infrastructure/event_camera/iterators/bias_events_iterator.py
from typing import Optional
import os.path
import metavision_hal as mv_hal
from metavision_core.event_io import EventsIterator
from src.domain.value_objects.bias_settings import BiasSettings
from src.application.interfaces.event_iterator import IEventIterator

class BiasEventsIterator(IEventIterator):
    def __init__(
        self,
        delta_t: int,
        input_filename: Optional[str] = None,
        bias_file: Optional[str] = None
    ):
        self.__is_live = not input_filename

        # Load bias settings if bias file is provided
        bias_settings = None
        if bias_file:
            bias_settings = BiasSettings.from_file(bias_file)
        elif self.__is_live:  # Use default settings for live camera
            bias_settings = BiasSettings.create_default()

        if not self.__is_live:
            if not os.path.exists(input_filename):
                raise FileNotFoundError(f"Input file not found: {input_filename}")
            self.__ev_it = EventsIterator(input_filename, delta_t=delta_t)
        else:
            self.device = mv_hal.DeviceDiscovery.open("")
            if not self.device:
                raise RuntimeError("No live camera found")

            # Configure camera biases
            if bias_settings:
                biases = self.device.get_i_ll_biases()
                for name, bias in bias_settings.biases.items():
                    biases.set(name, bias.value)

            self.__ev_it = EventsIterator(self.device, delta_t=delta_t)

    def __iter__(self):
        yield from self.__ev_it
        
    def __del__(self):
        if hasattr(self, '__ev_it'):
            self.__ev_it.close()  
        # if hasattr(self, 'device'):
        #     self.device.close()   

    def get_size(self) -> int:
        return self.__ev_it.get_size()