from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {'w1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'w2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
        self.lastLayer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        def loss_w(_):
            return self.loss(x, t)
        grads = {'w1': numerical_gradient(loss_w, self.params['w1']),
                 'b1': numerical_gradient(loss_w, self.params['b1']),
                 'w2': numerical_gradient(loss_w, self.params['w2']),
                 'b2': numerical_gradient(loss_w, self.params['b2'])}
        return grads

    def gradient(self, x, t):
        self.loss(x, t)
        d_out = 1
        d_out = self.lastLayer.backward(d_out)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_out = layer.backward(d_out)
        grads = {'w1': self.layers['Affine1'].dW,
                 'b1': self.layers['Affine1'].db,
                 'w2': self.layers['Affine2'].dW,
                 'b2': self.layers['Affine2'].db}
        return grads
