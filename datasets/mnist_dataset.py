import os
from typing import List
from typing import Callable
from enums.Datasets import DataSetType
import idx2numpy as idx2numpy
import numpy as np

"""Класс по считыванию данных"""


class MNISTDataset():
    # :param dataset_type (string): (['train', 'valid', 'test']).
    # :param transforms (list): список необходимых преобразований изображений.
    # :param nrof_classes (int): количество классов в датасете.
    # :param data_path (string): путь до файла с данными.

    def __init__(self,
                 dataset_type: DataSetType,
                 transforms: List[Callable],
                 nrof_classes: int,
                 dataset_path="./configs/mnist/",
                 images_file='train-images.idx3-ubyte',
                 labels_file='train-labels.idx1-ubyte'):
        self.__dataset_path = dataset_path
        self.__dataset_type = dataset_type
        self.__images_file = images_file
        self.__labels_file = labels_file
        self.__labels = []
        self.__images = []
        self.__transforms = transforms
        self.__nrof_classes = nrof_classes

        self.__stats = None

    def read_data(self):
        """
        Считывание данных по заданному пути+вывод статистики.
        """
        self.__labels = idx2numpy.convert_from_file(os.path.join(self.__dataset_path, self.__labels_file))
        self.__images = idx2numpy.convert_from_file(os.path.join(self.__dataset_path, self.__images_file))

        self.show_statistics()

    def __len__(self):
        """
        :return: размер выборки
        """

        return len(self.__images)

    def one_hot_labels(self, label):
        """
        для 10 классов метка 5-> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        res = []
        if (abs(label) > self.__nrof_classes):
            raise ValueError('Такого класса нет в датасете')

        for i in range(0, self.__nrof_classes):
            if (i == label):
                res.append(1)
            else:
                res.append(0)
        return res

    def __getitem__(self, idx: int):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        images = self.__images[idx]
        labels = self.__labels[idx]
        for transform in self.__transforms:
            images = transform(images)
        return images, labels

    def show_statistics(self):
        """
        Необходимо вывести количество элементов в датасете, количество классов и количество элементов в каждом классе
        """
        unique, counts = np.unique(self.__labels, return_counts=True)
        print(f'количество элементов в датасете: {self.__len__()}')
        print(f'количество классов: {self.__nrof_classes}')
        print(f'количество элементов в каждом классе : {dict(zip(unique, counts))}')