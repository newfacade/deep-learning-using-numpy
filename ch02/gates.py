import numpy as np


def _and(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def _not_and(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def _or(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def _xor(x1, x2):
    s1 = _not_and(x1, x2)
    s2 = _or(x1, x2)
    y = _and(s1, s2)
    return y


print(_and(0, 0))
print(_and(1, 0))
print(_and(0, 1))
print(_and(1, 1))
print(_not_and(0, 0))
print(_not_and(1, 0))
print(_not_and(0, 1))
print(_not_and(1, 1))
print(_or(0, 0))
print(_or(1, 0))
print(_or(0, 1))
print(_or(1, 1))
print(_xor(0, 0))
print(_xor(1, 0))
print(_xor(0, 1))
print(_xor(1, 1))

