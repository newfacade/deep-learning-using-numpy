import numpy as np
from common.functions import softmax, cross_entropy_error
from common.util import im2col, col2im


class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, d_out):
        d_out[self.mask] = 0
        return d_out


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, d_out):
        dx = d_out * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.dx = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, d_out):
        self.dx = np.dot(d_out, self.W.T)
        self.dW = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis=0)
        return self.dx


class SoftMaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, d_out=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class Convolution:
    def __init__(self, w, b, stride=1, pad=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_w = None
        self.db = None
        self.dw = None

    def forward(self, x):
        fn, c, fh, fw = self.w.shape
        n, c, h, w = x.shape
        out_h = int(1 + (h + 2 * self.pad - fh) / self.stride)
        out_w = int(1 + (w + 2 * self.pad - fw) / self.stride)
        col = im2col(x, fh, fw, self.stride, self.pad)
        col_w = self.w.reshape(fn, -1).T
        out = np.dot(col, col_w) + self.b
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.x = x
        self.col = col
        self.col_w = col_w
        return out

    def backward(self, d_out):
        fn, c, fh, fw = self.w.shape
        d_out = d_out.transpose(0, 2, 3, 1).reshape(-1, fn)
        self.db = np.sum(d_out, axis=0)
        self.dw = np.dot(self.col.T, d_out)
        self.dw = self.dw.transpose(1, 0).reshape(fn, c, fh, fw)
        d_col = np.dot(d_out, self.col_w.T)
        dx = col2im(d_col, self.x.shape, fh, fw, self.stride, self.pad)
        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = int(1 + (h - self.pool_h) / self.stride)
        out_w = int(1 + (w - self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        out = np.max(col, axis=1)
        out = out.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)
        self.x = None
        self.arg_max = None
        return out

    def backward(self, d_out):
        d_out = d_out.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        d_max = np.zeros((d_out.size, pool_size))
        d_max[np.arange(self.arg_max.size), self.arg_max.flatten()] = d_out.flatten()
        d_max = d_max.reshape(d_out.shape + (pool_size,))
        d_col = d_max.reshape(d_max.shape[0] * d_max.shape[1] * d_max.shape[2], -1)
        dx = col2im(d_col, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

