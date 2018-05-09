from .module import Module
from torch import Tensor

class Relu(Module):

    def __init__(self):
        super()

    def forward(self, x):
        self.x = x.clone()
        return x.clamp(min=0)

    def backward(self, dz):
        output = dz.clone()
        output[self.x < 0] = 0
        return output


class Tanh(Module):

    def __init__(self):
        super()

    def forward(self, x):
        self.x = x.clone()
        self.output = Tensor.tanh(x)
        return self.output;

    def backward(self, dz):
        return dz.mul(1.0 - Tensor.tanh(self.x).pow(2))
