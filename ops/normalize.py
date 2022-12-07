from ops.transformation import Transformation


class Normalize(Transformation):
    def __init__(self, mean=128, var=255):
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        :param var (int): значение, на которое необходимо поделить.
        """
        self.__mean = mean
        self.__var = var

    def __call__(self, images):
        return (images - self.__mean)/self.__var