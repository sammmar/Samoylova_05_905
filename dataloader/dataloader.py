# -*- coding: utf-8 -*-
"""Data Loader"""
"""Здесь находятся все классы и функции загрузки и предварительной обработки данных. При желании можно разделить"""
import torch, torchvision
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt


class DataLoader:
    """Data Loader class"""

    def __init__(self, dataset, dataset_type, batch_size,nrof_classes,
                 sample_type,
                 epoch_size=None,
                 shuffle=True):
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__sample_type = sample_type
        self.__epoch_size = epoch_size
        self.__dataset_type = dataset_type
        self.__nrof_classes = nrof_classes


    def batch_generator(self):
        """
        Создание батчей на эпоху с учетом указанного размера эпохи и типа сэмплирования.
        """
        indexes = np.arange(len(self.__dataset))
        if (self.__shuffle):
            np.random.shuffle(indexes)
        for i in range(0, len(indexes), self.__batch_size):
            self.__batch = self.__dataset[indexes[i: i + self.__batch_size]]
            yield self.__batch

    def show_batch(self):
        """
        Необходимо визуализировать и сохранить изображения в батче (один батч - одно окно). Предварительно привести значение в промежуток
        [0, 255) и типу к uint8
        :return:
        """
        img, label = self.__batch

        for image in img:
            image = np.array(image, dtype='uint8')
            image = image.reshape((28, 28))

        pict = plt.subplots(figsize=(5, 3))
        # Поочередно считываем в переменную picture имя изображения из списка img . В переменную i записываем номер итерации
        for i, picture in enumerate(img):
            pict.add_subplot(
                int(sqrt(len(img))),
                int(sqrt(len(img))),
                i + 1)
            plt.imshow(picture)
            plt.title(label[i])
            plt.axis('off')

        plt.show()





