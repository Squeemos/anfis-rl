import yaml

class Proxy(object):
    def __init__(self, attributes):
        super().__init__()

        for key, value in attributes.items():
            setattr(self, key, value)

class Config(object):
    def __init__(self, path):
        super().__init__()

        with open(path) as stream:
            config = yaml.safe_load(stream)

        for key, value in config.items():
            setattr(self, key, Proxy(value))
