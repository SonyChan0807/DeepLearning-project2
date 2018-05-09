from .module import Module
from .linear import Linear


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
