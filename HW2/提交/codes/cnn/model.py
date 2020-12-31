# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features # channel_size
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Parameters
		self.eps = 1e-5
		self.momentum = 0.1
		self.weight = Parameter(torch.ones((1, num_features, 1, 1), dtype=torch.float), requires_grad=True)
		self.bias = Parameter(torch.zeros((1, num_features, 1, 1), dtype=torch.float), requires_grad=True)

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(size=(1, num_features, 1, 1), dtype=torch.float).to(self.device))
		self.register_buffer('running_var', torch.ones(size=(1, num_features, 1, 1), dtype=torch.float).to(self.device))

		# Initialize your parameter

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		n, c, h, w = input.size()
		if self.training:
			input_mean = torch.mean(input, dim=(0, 2, 3), keepdim=True).to(self.device)
			input_var = torch.var(input, dim=(0, 2, 3), keepdim=True, unbiased=False).to(self.device)
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
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			mask = torch.rand(input.size(), device=input.device) > self.p
			input = (input * mask) / (1.0 - self.p)
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3 , padding=1),
								BatchNorm2d(num_features=128),
								nn.ReLU(),
								Dropout(drop_rate),
								nn.MaxPool2d(kernel_size=2),
								nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
								BatchNorm2d(num_features=256),
								nn.ReLU(),
								Dropout(drop_rate),
								nn.MaxPool2d(kernel_size=2),
								)
		self.linear = nn.Linear(in_features=256 * 8 * 8, out_features=10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		x1 = self.conv1(x)
		n, c, h, w = x1.size()
		x2 = x1.reshape(n, c*h*w)
		logits = self.linear(x2)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y.long())
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
