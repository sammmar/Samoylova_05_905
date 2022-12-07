from asyncio.windows_events import NULL
from utils.Config.BaseConfig import BaseConfig

class Model(BaseConfig):
    """Класс-модель "Model" для определения полей конфига"""

    __input = []
    """"""

    __up_stack = {}
    """Слои"""

    __acivation_function = NULL
    """Функция активации"""

    __output = NULL
    """Кол-во нейронов на выходе"""

    @property
    def input(self):
        return self.__input

    @property
    def up_stack(self):
        return self.__up_stack

    @property
    def acivation_function(self):
        return self.__acivation_function

    @property
    def output(self):
        return self.__output

    def __init__(self,dataModel):
        super().__init__(dataModel)

    def _parse_xml_to_datamodel(self,datamodel):
        self.__acivation_function = datamodel.find('acivation_function').text
        self.__output = datamodel.find('output').text

        for up_stacks in datamodel.find('up_stack'):
            self.__up_stack.update({up_stacks.tag : up_stacks.text})
        for inputs in datamodel.findall('input'):
            self.__input.append(inputs.text)