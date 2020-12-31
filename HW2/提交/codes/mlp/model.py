# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Parameters
		self.eps = 1e-5
		self.momentum = 0.1
		self.weight = Parameter(torch.ones((num_features,), dtype=torch.float), requires_grad=True)
		self.bias = Parameter(torch.zeros((num_features,), dtype=torch.float), requires_grad=True)

		# Store the average mean and variance

		self.register_buffer('running_mean', torch.zeros(size=(num_features,), dtype=torch.float).to(self.device))
		self.register_buffer('running_var', torch.ones(size=(num_features,), dtype=torch.float).to(self.device))

		# Initialize your parameter


	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			input_mean = input.mean(axis=0)
			input_var = input.var(axis=0)
			input_norm = (input - input_mean) / torch.sqrt(input_var + self.eps)
			output = self.weight * input_norm + self.bias

			self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * input_mean
			self.running_var = (1-self.momentum) * self.running_var + self.momentum * input_var
		else:
			output = self.weight * (input - self.running_mean) / torch.sqrt(self.running_var + self.eps) + self.bias
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mask = torch.rand(input.size(), device=input.device) > self.p
			input = (input * mask) / (1 - self.p)
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.linear = nn.Sequential(nn.Linear(3 * 32 * 32, 512),
									BatchNorm1d(512),
									nn.ReLU(),
									Dropout(drop_rate),
									nn.Linear(512, 10))

		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.linear(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y.long())
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
