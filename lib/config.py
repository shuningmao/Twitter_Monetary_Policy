import json


class ModelConfig(object):
    def __str__(self):
        return str(self.__dict__)


class Config(object):

    def __init__(self, fname):
        train = None
        prod = None
        with open(fname, 'rb') as f:
            conf = json.load(f)

            train = ModelConfig()
            prod = ModelConfig()

            train.__dict__ = conf['train']
            prod.__dict__ = conf['train'].copy()
            prod.__dict__.update(conf['prod'])

            train.train = True
            prod.train = False

        self.train = train
        self.prod = prod
