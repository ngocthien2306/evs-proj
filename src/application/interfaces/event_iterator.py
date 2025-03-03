from abc import ABC, abstractmethod
from typing import Iterator, Any

class IEventIterator(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass