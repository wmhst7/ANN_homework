from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        return np.sum((target - input) ** 2) / (2.0 * input.shape[0])
        # TODO END

    def backward(self, input, target):
        # TODO START
        # print('Loss Backrward: ', target.shape)
        return (input - target) / input.shape[0]
        # TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        inexp = np.exp(input)
        pk = inexp / np.sum(inexp, axis=1, keepdims=True)
        Ek = -np.sum(target * np.log(pk)) / input.shape[0]
        self.softmax = pk
        return Ek
        # TODO END

    def backward(self, input, target):
        # TODO START
        pk = self.softmax
        return (pk / np.sum(target, axis=1, keepdims=True) - target) / input.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, threshold=0.05):
        self.name = name

    def forward(self, input, target):
        # TODO START
        Delta = 5
        x = np.max(np.where(target == 1, input, 0.), axis=1, keepdims=True)
        a = np.array(np.maximum(0., Delta - x + input))
        b = np.where(target == 1, 0., a)
        res = np.sum(b) / input.shape[0]
        self.b = b
        return res
        # TODO END

    def backward(self, input, target):
        # TODO START
        b = self.b
        c = np.where(b > 0., 1., 0.)
        m = np.zeros(input.shape) - np.sum(c, axis=1, keepdims=True)
        res = np.where(target == 1, m, c) / input.shape[0]
        return res
        # TODO END

