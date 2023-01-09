import yaml

class Config(object):
    def __init__(self, path):
        super().__init__()

        with open(path) as stream:
            config = yaml.safe_load(stream)

        for key, value in config.items():
            setattr(self, key, value)

    def __str__(self):
        max_len = len(max(self.__dict__, key=len)) + 1
        output = ""
        for i, (k, v) in enumerate(self.__dict__.items()):
            output += f"{k:{max_len}}: {v}"
            if i + 1 != len(self.__dict__):
                output += "\n"
        return output
