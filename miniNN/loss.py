
import torch
from torch import Tensor


class Loss(object):

    def forward(self, x, t):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSELoss(Loss):

    def __init__(self):
        super()

    def forward(self, x, t):
        self.t = t.clone()
        self.x = x.clone()
        self.output = (self.x - self.t).pow(2).mean()

        return self.output

    def backward(self):
        dloss = (self.x - self.t) / self.x.shape[0]
        return dloss