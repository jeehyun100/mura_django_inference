from .DenseNet import DenseNet169
from ..config import opt
import os



class MuraModel:
    def __init__(self):
        self.model = DenseNet169()
        opt.parse({})
        self.opt = opt


    def load(self):
        print("checkpoint file {0}".format(self.opt.load_model_path))
        #print(os.path.abspath(__file__))
        self.model.load(self.opt.load_model_path)
        print("model load done")
        return self.model

