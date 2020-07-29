from common.layers import *
from collections import OrderedDict


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param=None, hidden_size=100, output_size=10, weight_init_std=0.01):
        if conv_param is None:
            conv_param = {"filter_num":30, "filter_size": 5, "pad": 0, "stride": 1}
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = int((input_size - filter_size + 2 * filter_pad) / filter_stride + 1)
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        self.params = {'w1': weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size),
                       'b1': np.zeros(filter_num),
                       'w2': weight_init_std * np.random.randn(pool_output_size, hidden_size),
                       'b2': np.zeros(hidden_size), 'w3': weight_init_std * np.random.randn(hidden_size, output_size),
                       'b3': np.zeros(output_size)}
        self.layers = OrderedDict()
        self.layers['conv1'] = Convolution(self.params['w1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['relu1'] = ReLu()
        self.layers['pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['affine1'] = Affine(self.params['w2'], self.params['b2'])
        self.layers['relu2'] = ReLu()
        self.layers['affine2'] = Affine(self.params['w3'], self.params['b3'])
        self.last_layer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)
        d_out = 1
        d_out = self.last_layer.backward(d_out)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_out = layer.backward(d_out)
        grads = {'w1': self.layers['conv1'].dw, 'b1': self.layers['conv1'].db, 'w2': self.layers['affine1'].dw,
                 'b2': self.layers['affine1'].db, 'w3': self.layers['affine2'].dw, 'b3': self.layers['affine2'].db}
        return grads
