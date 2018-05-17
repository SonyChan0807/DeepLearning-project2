
import torch
from torch import Tensor


class Loss(object):
    """Abstract class of loss function
    Every new loss function inherits Loss should implement forward and backward.

    """

    def forward(self, x, t):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSELoss(Loss):

    def __init__(self):
        super()

    def forward(self, x, t):
        """Forward pass for MSE loss
         Compute mean squared error

        :math: ` \frac{1}{n} \Sigma {(X - T)^2}`

        :param x: Output tensor from neural network
        :param t: Data label

        :return: value of Mean-squared Loss
        """
        self.t = t.clone()
        self.x = x.clone()
        output = (self.x - self.t).pow(2).mean()

        return output

    def backward(self):
        """Backward pass for MSE loss
        Compute the derivatives of the loss wrt input x

        :math: ` \frac{1}{n} {(X - T)}`

        :return: The derivatives of the loss wrt input x
        """

        dloss = (self.x - self.t) / self.x.shape[0]
        return dloss