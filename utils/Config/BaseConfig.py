from abc import ABC, abstractmethod

class BaseModelConfig(ABC):


    def __init__(self, datamodel):
        self._parse_xml_to_datamodel(datamodel)

    @abstractmethod
    def _parse_xml_to_datamodel(self,datamodel):
        pass