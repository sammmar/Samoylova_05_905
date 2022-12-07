class Registry(object):
    def __init__(self, name):
        self.name = name
        self._module_dict = dict()

    def get(self, key, kwargs):
        return self._module_dict.get(key, None)(**kwargs)

    def _register_module(self, module_class):
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls

REGISTRY_TYPE = Registry('Layers')
