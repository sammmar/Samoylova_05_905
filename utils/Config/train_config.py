from asyncio.windows_events import NULL
from utils.Config.BaseConfig import BaseConfig

class Train(BaseConfig):
    """Класс-модель "Train" для определения полей конфига"""

    __batch_size = NULL
    """Размер батчей"""

    __nrof_epoch = NULL
    """Кол-во эпох"""

    __optimizer = {}
    """Оптимизатор"""

    __metrics = []
    """Метрики"""

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def nrof_epoch(self):
        return self.__nrof_epoch

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def metrics(self):
        return self.__metrics

    def __init__(self,dataModel):
        super().__init__(dataModel)

    def _parse_xml_to_datamodel(self,datamodel):
        self.__batch_size = int(datamodel.find('batch_size').text)
        self.__nrof_epoch = int(datamodel.find('nrof_epoch').text)

        for optimizer in datamodel.find('optimizer'):
            self.__optimizer.update({optimizer.tag : optimizer.text})
        for metrics in datamodel.findall('metrics'):
            self.__metrics.append(metrics.text)