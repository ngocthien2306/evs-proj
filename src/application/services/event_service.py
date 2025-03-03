from typing import Optional, Union
from ...domain.value_objects.bias_settings import BiasSettings
from ..interfaces.event_iterator import IEventIterator
from ...infrastructure.event_camera.iterators.buffered_iterator import BufferedEventIterator
from ...infrastructure.event_camera.iterators.non_buffered_iterator import NonBufferedEventIterator
from ...infrastructure.event_camera.utils.bias_file_loader import load_bias_file

class EventService:
    @staticmethod
    def create_iterator(
        delta_t: float,
        input_filename: Optional[str] = None,
        bias_file: Optional[str] = None,
        buffered: bool = True
    ) -> IEventIterator:
        bias_settings = None
        if bias_file:
            bias_settings = load_bias_file(bias_file)
        elif not input_filename:  # Live camera with default settings
            bias_settings = BiasSettings.create_default()

        Iterator = BufferedEventIterator if buffered else NonBufferedEventIterator
        return Iterator(delta_t, input_filename, bias_settings)