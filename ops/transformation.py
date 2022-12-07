from abc import ABC, abstractmethod
import numpy as np

class Transformation(ABC):

    @abstractmethod
    def __call__(self, images):
        pass
