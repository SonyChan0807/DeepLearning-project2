from .module import Module
from torch import Tensor

class Relu(Module):

    def __init__(self):
        super()

    def forward(self, x):
        """Applies the rectified function to the input  element-wise

        :param x: Input tensor from Linear modules

        :return: tensor
        """
        self.x = x.clone()
        return x.clamp(min=0)

    def backward(self, dz):
        """Backward pass of ReLu
        Compute the derivatives of the loss wrt the activations

        :math `\nabla_{x^{(l)}}  \ell  * x.sign()`

        :param dz: The gradient output from the previous layer

        :return: The derivatives of the loss wrt RuLu
        """
        output = dz.clone()
        output[self.x < 0] = 0
        return output


class Tanh(Module):

    def __init__(self):
        super()

    def forward(self, x):
        """Applies Tanh  to the input element-wise

        :param x: Input tensor from Linear modules

        :return: tensor
        """
        self.x = x.clone()
        self.output = Tensor.tanh(x)
        return self.output;

    def backward(self, dz):
        """ Backward pass of Tanh
        Compute the derivatives of the loss wrt the activations

        :math `\nabla_{x^{(l)}}  \ell  * (1 - tanh(s^(l))^2)`

        :param dz: The gradient output from the  liner layer

        :return:  The derivatives of the loss wrt Tanh
        """
        return dz.mul(1.0 - Tensor.tanh(self.x).pow(2))
