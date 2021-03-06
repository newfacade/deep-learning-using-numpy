from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {'w1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'w2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
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
