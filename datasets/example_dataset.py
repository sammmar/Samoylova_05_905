import torchvision
from ast import List
from typing import List
import json
from typing import Callable
from enums.Datasets import DataSetType
import numpy as np
import idx2numpy
"""здесь пишем класс по считыванию данных, удобнее для каждого набора данных выделять отдельный файл и класс"""

class Dataset:
    def __init__(self, dataset_type: DataSetType, transforms: List[Callable],
    nrof_classes: int,
    dataset_path = 'C:\\Users\\sambs\\Desktop\\',
                 img_path='%s/train-images.idx3-ubyte' % dataset_path,
                lbl_path = '%s/train-labels.idx1-ubyte' % dataset_path):
        self.__dataset_path = dataset_path
        self.__dataset_type = dataset_type
        self.__img_path = img_path
        self.__lbl_path = lbl_path
        self.__labels = []
        self.__images = []
        self.__transforms = transforms
        self.__nrof_classes = nrof_classes

        """
        :param data_path (string): путь до файла с данными.
        :param dataset_type (string): (['train', 'valid', 'test']).
        :param transforms (list): список необходимых преобразований изображений.
        :param nrof_classes (int): количество классов в датасете.
        """

    def read_data(self):
        """
                Считывание данных по заданному пути+вывод статистики.
                """

        self.__images = idx2numpy.convert_from_file(img_path)
        self.__labels = idx2numpy.convert_from_file(lbl_path)

        self.show_statistics()

    def __len__(self):
        """
        :return: размер выборки
        """
        len_img=len(self.__images)
        return len_img




    def one_hot_labels(self, label):
        """
        для 10 классов метка 5-> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        encode_one_hot = []

        for lbl in range(0, self.__nrof_classes):
            if (lbl == label):
                encode_one_hot.append(1)
            else:
                encode_one_hot.append(0)
        return encode_one_hot

    def __getitem__(self, idx):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        images = self.images[idx]
        labels = self.labels[idx]
        for transform in self.transforms:
            images = transform(images)
        return images, labels

    def show_statistics(self):
        """
        Необходимо вывести количество элементов в датасете, количество классов и количество элементов в каждом классе
        """

        print('Количество элементов в датасете:',self.__len__())
        print('Количество классов:',self.__nrof_classes)



