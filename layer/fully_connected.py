import math
import numpy as np
from layers.registry_utils import REGISTRY_TYPE
from layers.base_layer import BaseLayerClass


REGISTRY_TYPE = Registry('Layers')


@REGISTRY_TYPE.register_module
class FullyConnected(BaseLayerClass):
    def __init__(self, input_size, output_size):
        self.bias = np.zeros_like(self.x)
        self.weight = np.random.normal(0, math.sqrt(len(self.x)), size = (input_size, output_size))


    def __call__(self, x, phase):
        self.x = x
        out = np.exp(-x)
        return 1/(1+out)

    @property
    def trainble(self):
        return True

    def get_grad(self):
        out = np.exp(- self.x)
        self.grads = out/(1+out)**2
        return self.grads

    def backward(self, dy): # градиенты по текущему классу относительно целевой функции
        self.get_grad()
        self.x_grad = np.dot(self.grads[0], dy.T)
        self.weight_grad = np.mean(self.gards[2] * dy, 0)
        return self.x_grad.T


    def update_weigfts(self, update_func):
        ...

