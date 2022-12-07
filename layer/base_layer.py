from abc import abstractmethod
import numpy as np


class BaseLayerClass(object):
    def __init__(self):
        """
        инициализация обучаемых параметров нейронной сети случайным образом по формуле из презентации
        """
        ...

    @abstractmethod
    def __call__(self, x, phase):
        return x

    @property
    def trainable(self):
        """
        свойство, является ли слой обучаемым
        """
        return False

    def get_grad(self):
        """
        вычисление градиентов по этому слою
        """
        pass

    @abstractmethod
    def backward(self, dy):
        """
        цепное правило дифференцирования
        :param dy: значение градиента пришедшего от следующего слоя
        :return: значение градиента этого слоя
        """
        pass

    def update_weights(self, update_func):
        """
        обновление обучаемых параметров, если они есть, иначе ничего
        :param update_func: функция обновления, указано в презентации
        """
        pass

    def get_nrof_trainable_params(self):
        """
        вычисление количества обучаемых параметров
        :return: количество обучаемых параметров
        """
        return 0

