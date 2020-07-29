import numpy as np
from dataset.mnist import load_mnist
import pickle
from common.functions import sigmoid, softmax


def get_data():
    (_, _), (x_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, y_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y


x, t = get_data()
net = init_network()
accuracy = 0
batch_size = 100
for i in range(0, len(x), batch_size):
    x_batch = x[i: i+batch_size]
    y_batch = predict(net, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy += np.sum(p == t[i: i+batch_size])

print(accuracy/len(x))
