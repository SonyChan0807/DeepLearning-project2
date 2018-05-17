from .module import Module
from .linear import Linear


class Sequential(Module):

    def __init__(self, layer_modules):
        """Sequential model of neural network

        :param layer_modules: List of models that construct neural network
        """
        super()
        self.layer_modules = layer_modules

    def forward(self, x_input):
        """Forward pass for neural network
        Compute the forward pass for each layers and activation function

        :param x_input: input tensor of data

        :return: The result of neural network to the loss function
        """
        module_input = x_input.clone()

        for i in range(len(self.layer_modules)):
            module_output = self.layer_modules[i].forward(module_input)
            module_input = module_output

        return module_output

    def backward(self, dz):
        """Backward pass for neural network
        Compute the backward pass for each layers and activation function

        :param dz:  The derivatives of loss wrt neural network output from the Loss class

        :return: None
        """
        for m in self.layer_modules[::-1]:
            dz = m.backward(dz)

    def update_params(self, lambda_):
        """Update weight and bias in each Linear class

        :param lambda_: learning rate

        :return: None
        """
        for m in self.layer_modules:
            if isinstance(m, Linear):
                m.update_params(lambda_)

    def zero_gradient(self):
        """Set weight and bias to zero in each Linear class

        :return: None
        """
        for m in self.layer_modules:
            if isinstance(m, Linear):
                m.zero_gradient()

    def params(self):
        """Get params of each linear layer

        :return: List of weight and bias in the same order of the linear modules
        """
        param_list = []

        for m in self.layer_modules:
            if isinstance(m, Linear):
                param_list.append(m.param)

        return param_list
