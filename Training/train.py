import numpy as np
from tensorboardX import SummaryWriter
from configs.config import CFG
from dataloader.dataloader import DataLoader
from datasets.mnist_dataset import MNISTDataset
from ops.normalize import Normalise
from ops.view import View
from enums.Datasets import DataSetType
from model.cross_entripy_loss import CrossEntripyLoss
from model.sgd import SGD

class Trainer:
    def __init__(self, cfg):
        self.net = Model(CFG['model']['parametrs'])
        self.loss = CrossEntripyLoss()
        self.optimiser = SGD(CFG['train']['learning_rate'], self.net, self.loss)
        self.epochs = CFG['train']['nrof_epoch']
        self.train_dataset = MNISTDataset(dataset_type=DataSetType.train,
                                          transforms=[Normalise(), View()],
                                          nrof_classes=cfg.data.nrof_classes)
        self.test_dataset = MNISTDataset(dataset_type=DataSetType.test,
                                         transforms=[Normalize(), View()],
                                         nrof_classes=cfg.data.nrof_classes,
                                         image_file='t10k-images.idx3-ubyte',
                                         labels_file='t10k-labels.idx3-ubyte'
                                         )
        self.test_dataloader = DataLoader(self.test_dataset,
                                          cfg.train.batch_size,
                                          None,
                                          cfg.train.nrof_epoch)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           cfg.train.batch_size,
                                           None,
                                           cfg.train.nrof_epoch)
        self.writer = SummaryWriter("mlp")
        self.global_iteration = 0


    def fit(self):
        """
        функция обучения нейронной сети, включает в себя цикл по эпохам, в рамках одной эпохи:
        вызов train_epoch;
        вызов evaluate для обучающей и валидационной выборки последовательно
        """
        for epoch in range (self.epochs):
            self.train_epoch()
            self.evaluate(self.train_dataloader)
            self.evaluate(self.test_dataloader)



    def evaluate(self, dataloader, dataset_type):
        """
        функция вычисления значения целевой функции и точности
        на всех данных из dataloader, сохранение результатов в tensorboard/mlflow/plotly
        :param dataloader: train_loader/valid_loader/test_loader
        """
        batch_loss_list = [] #целевая функция на каждой итерации
        batch_accuracy_list = [] #точность на каждой итерации
        batch_size_list = [] #размер батча на каждой итерации
        total_loss = np.sum(batch_loss_list*batch_size_list)/np.sum(batch_size_list) #среднее значение целевой функции
        total_accuracy = np.sum(batch_accuracy_list*batch_size_list)/np.sum(batch_size_list)
        self.writer.add_scalar(dataset_type + "/total_loss", total_loss, self.global_iteration)

    def train_epoch(self):
        """
        функция обучения нейронной сети: для каждого батча из обучающей
        выборки вызывается метод _train_step и логируется значение целевой функции и точности на батче
        """
        self.net.train()
        for batch, labels in self.train_dataloader.batch_generator():
            self._train_step(batch, labels)
        self.net.eval()

    def _train_step(self):
        """
        один шаг обучения нейронной сети, в рамках которого происходит:
        1) вызов forward pass нейронной сети для батча;
        2) вычисление значения целевой функции;
        3) вызов функции minimize для вычисления обратного распространения и обновления весов
        """
        logits = self.net(batch)
        batch_loss = self.loss(logits, labels)
        batch_accuracy = (np.argmax(logits, 1) == labels).mean
        self.optimiser.minimize()
        self.writer.add_scalar('batch_loss', batch_loss, self.global_iteration)
        self.writer.add_scalar('batch_accuracy', batch_accuracy, self.global_iteration)
        self.global_iteration+=1
        return batch_loss, batch_acurasy


    def overfit_on_batch(self):
        batch, labels = next(self.train_dataloader.batch_generator())
        self.net.train()
        for i in range(1000):
            batch_loss, batch_acurasy = self._train_step(batch, labels)
            print(batch_loss, batch_acurasy)
            # значение точности на протяжении 10 шагов подряд была 100%
        self.net.eval()

