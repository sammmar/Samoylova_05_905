class BaseModel(object):
    def __init__(self, parameters):
        ...

    def eval(self):
        self.phase = 'eval'

    def train(self):
        self.phase = 'train'

    def __call__(self, input):
        """
        цикл по всем слоям нейронной сети и вызов метода forward pass (__call__)
        :param input: батч значений предыдущего слоя
        :return: значения текущего слоя
        """

    def get_parameters(self):
        """
        получение dict, где в качестве key — название слоя, в качестве value — значения обучаемых параметров слоя
        :return:
        """

    def dump_model(self, path):
        """
        функция сохранения параметров нейронной сети в pickle-файл
        :param path: путь, куда сохранять модель
        """

    def load_weights(self, path):
        """
        функция считывания параметров нейронной сети из pickle-файла
        :param path: путь, откуда считывать модель
        """