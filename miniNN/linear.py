import math
from .module import Module
from torch import Tensor


class Linear(Module):

    def __init__(self, input_size, hidden_nodes):
        """Applies a linear transformation to input data

        :param input_size:  size of each input sample
        :param hidden_nodes: size of each output sample / number of hidden nodes.
        """
        super()

        # PyTorch like initialization
        stdv = 1. / math.sqrt(input_size)
        self.w = Tensor(hidden_nodes, input_size).uniform_(-stdv, stdv)
        self.b = Tensor(hidden_nodes).uniform_(-stdv, stdv)
        self.dw = Tensor(self.w.size()).zero_()
        self.db = Tensor(self.b.size()).zero_()

    def forward(self, x):
        """Forward pass for full connected layer

        :math `s = x \cdot w^T + b `

        :param x: input Tensor

        :return: s:  Tensor after transformation
        """
        self.x = x.clone()
        s = x.matmul(self.w.t()) + self.b

        return s

    def backward(self, dz):
        """ Backward pass for full connected layer
        Compute the derivatives of the loss wrt the parameters

        :math  `
            \nabla_{x^{(l)}}  \ell  =  \nabla_{s^{(l)}} (w^{l+1})
            \nabla_{w^{(l)}}  \ell  =  \nabla_{s^{(l)}}  \ell (x^{l-1})^T
            \nabla_{b^{(l)}}  \ell  =  \nabla_{s^{(l)}}  \ell
         `
        :param dz: the gradient output from activation function
        
        :return: The derivatives of the loss wrt input x
        """
        dx = dz.matmul(self.w)
        self.dw += dz.t().matmul(self.x)
        self.db += dz.t().sum(1)
        return dx

    def param(self):
        """Get weight and bias

        :return: param_list: list of weight and bias
        """
        param_list = [self.w, self.b]

        return param_list

    def update_params(self, lambda_):
        """Update weight and bias

        :param lambda_: learning rate

        :return None
        """

        self.w -= lambda_ * self.dw
        self.b -= lambda_ * self.db

    def zero_gradient(self):
        """Set weight and bias to zero,

        :return: None
        """
        self.dw.zero_()
        self.db.zero_()

