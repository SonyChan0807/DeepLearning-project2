
import math
from .module import Module
from torch import Tensor


class Linear(Module):

    def __init__(self, input_size, hidden_nodes):
        super()
        # Initialize weight, bias xavie initializer
        stdv = 1. / math.sqrt(input_size)
        self.w = Tensor(hidden_nodes, input_size).uniform_(-stdv, stdv)
        self.b = Tensor(hidden_nodes).uniform_(-stdv, stdv)
        self.dw = Tensor(self.w.size()).zero_()
        self.db = Tensor(self.b.size()).zero_()

    def forward(self, x):
        self.x = x.clone()
        s = x.matmul(self.w.t()) + self.b
        return s

    def backward(self, dz):
        dx = dz.matmul(self.w)
        self.dw += dz.t().matmul(self.x)
        self.db += dz.t().sum(1)
        return dx

    def param(self):
        param_list = [self.w, self.b]
        return param_list

    def update_params(self, lambda_):
        self.w -= lambda_ * self.dw
        self.b -= lambda_ * self.db

    def zero_gradient(self):
        self.dw.zero_()
        self.db.zero_()

