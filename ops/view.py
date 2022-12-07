import numpy as np
from ops.transformation import Transformation


class View(Transformation):
    def __init__(self):
        """
        reshape image to vector
        """
        pass

    def __call__(self, images):
        return images.reshape((images.shape[0], -1))