import numpy as np


def step_function(t):
    return np.array(t > 0, dtype=int)


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 防止溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def mean_square_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return 0.5 * np.sum((y - t) ** 2) / batch_size


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

