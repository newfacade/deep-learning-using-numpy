import numpy as np


class DropOut:
    def __init__(self, dropout_ratio=0.7):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag=True):
        if train_flag:
            self.mask = np.random.randn(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, d_out):
        return d_out * self.mask

