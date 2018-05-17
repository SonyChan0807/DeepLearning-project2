import torch
from torch import Tensor


class Module(object):
    """Abstract class of loss function
    Every new module inherits Module should implement forward and backward, param is optional.

    """

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dz):
        raise NotImplementedError

    def param(self):
        return []






