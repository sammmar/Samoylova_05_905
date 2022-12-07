from asyncio.windows_events import NULL
from utils.Config.BaseConfig import BaseConfig

class Data(BaseConfig):
    """Класс-модель "Data" для определения полей конфига"""

    __path = NULL
    """Путь к датасету"""
    __image_size = NULL
    """Размерность изображения"""
    __load_with_info = NULL
    """Подгружать ли доп информацию"""
    __nrof_classes = NULL
    """Количество классов"""
    __shuffle = NULL

    @property
    def path(self):
        return self.__path

    @property
    def image_size(self):
        return self.__image_size

    @property
    def load_with_info(self):
        return self.__load_with_info

    @property
    def nrof_classes(self):
        return self.__nrof_classes

    @property
    def shuffle(self):
        return self.__shuffle;

    def __init__(self,dataModel):
        super().__init__(dataModel)

    def _parse_xml_to_datamodel(self,datamodel):
        self.__path = datamodel.find('path').text
        self.__image_size = datamodel.find('image_size').text
        self.__load_with_info = datamodel.find('load_with_info').text
        self.__nrof_classes = datamodel.find('nrof_classes').text
        self.__shuffle = datamodel.find('shuffle').text