from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

    def backward(self, input, target):
        '''Your codes here'''
        pass


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END


class HingeLoss(object):
	def __init__(self, name, threshold=0.05):
		self.name = name

	def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

	def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

