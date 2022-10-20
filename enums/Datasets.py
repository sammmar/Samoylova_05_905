from enum import Enum


class AvailableDatasets(Enum):
    MNIST = 'MNIST'
    KMNIST = 'KMNIST'

class DataSetType(Enum):
    Valid = 10
    Train = 20
    Test = 30