import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


net = SimpleNet()
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
# 改变net.W会改变net.loss!
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)
print(net.loss(x, t))
