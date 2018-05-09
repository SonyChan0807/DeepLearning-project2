import torch
from torch import Tensor


class Module(object):

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dz):
        raise NotImplementedError

    def param(self):
        return []






