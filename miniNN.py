import torch
from torch import Tensor
import math


class Module(object):

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dz):
        raise NotImplementedError

    def param(self):
        return []


class Loss(object):

    def forward(self, x, t):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


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


class Sequential(Module):

    def __init__(self, layer_modules):
        super()
        self.layer_modules = layer_modules

    def forward(self, x_input):
        module_input = x_input.clone()

        # hidden layer
        for i in range(len(self.layer_modules)):
            module_output = self.layer_modules[i].forward(module_input)
            module_input = module_output
        return module_output

    def backward(self, dz):
        for m in self.layer_modules[::-1]:
            dz = m.backward(dz)

    def update_params(self, lambda_):
        for m in self.layer_modules:
            if isinstance(m, Linear):
                m.update_params(lambda_)

    def zero_gradient(self):
        for m in self.layer_modules:
            if isinstance(m, Linear):
                m.zero_gradient()

    def params(self):
        param_list = []
        for m in self.layer_modules:
            if isinstance(m, Linear):
                param_list.append(m.param)
        return param_list

