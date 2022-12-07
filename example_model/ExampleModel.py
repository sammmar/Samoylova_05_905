import torch

from example_model.example_base_model import BaseModel
from nets.MLP import MLP
from dataloader.dataloader import DataLoader


class ExampleModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self):
        self.model = MLP(self.cfg['model'])
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()


    def load_data(self):
        self.dataset, self.info = DataLoader().load_data(self.config.mnist)

    def train(self):
        for data in self.dataset['train']:
            inputs, labels = data
            inputs = inputs.reshape(inputs.size(0), -1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
