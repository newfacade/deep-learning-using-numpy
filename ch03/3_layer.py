import numpy as np
from common.functions import sigmoid


def init_network():
    network = {'w1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]), 'b1': np.array([0.1, 0.2, 0.3]),
               'w2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]), 'b2': np.array([0.1, 0.2]),
               'w3': np.array([[0.1, 0.3], [0.2, 0.4]]), 'b3': np.array([0.1, 0.2])}
    return network


def forward(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = a3
    return y


net = init_network()
x = np.array([1.0, 0.5])
y = forward(net, x)
print(y)
