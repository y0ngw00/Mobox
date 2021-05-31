import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import filter

import time

def xavier_initializer(gain=1.0):
	def initializer(tensor):
		torch.nn.init.xavier_uniform_(tensor,gain=gain)
	return initializer
def uniform_initializer(lo=-1.0,up=1.0):
	def initializer(tensor):
		torch.nn.init.uniform_(tensor,a=lo,b=up)
	return initializer
	
class SlimFC(nn.Module):
	def __init__(self,
				 in_size,
				 out_size,
				 initializer,
				 activation=None):
		super(SlimFC, self).__init__()
		layers = []
		linear = nn.Linear(in_size, out_size)
		initializer(linear.weight)
		nn.init.constant_(linear.bias, 0.0)
		layers.append(linear)
		if activation == "relu":
			layers.append(nn.ReLU())
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)

class AppendLogStd(nn.Module):
	def __init__(self, init_log_std, dim , fixed_grad = True):
		super().__init__()
		self.log_std = torch.nn.Parameter(
			torch.as_tensor([init_log_std] * dim).type(torch.float32))
		self.register_parameter("log_std", self.log_std)
		self.log_std.requires_grad = not fixed_grad
	def set_value(self, val):

		self.log_std.data = torch.full(self.log_std.shape,np.log(val),device=self.log_std.device)
	def forward(self, x):
		x = torch.cat([x, self.log_std.unsqueeze(0).repeat([len(x), 1])], axis=-1)
		return x

class FC(nn.Module):
	def __init__(self, dim_in, model_config):
		nn.Module.__init__(self)

		hiddens = model_config['hiddens']
		activations = model_config['activations']
		init_weights = model_config['init_weights']
		
		layers = []
		prev_layer_size = dim_in
		
		for size, activation, init_weight in zip(hiddens + [1], activations, init_weights):
			# if size == 1:
			# 	layers.append(SlimFC(
			# 	prev_layer_size,
			# 	size,
			# 	uniform_initializer(-init_weight, init_weight),
			# 	activation))
			# else:
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation))
			prev_layer_size = size

		self.fn = nn.Sequential(*layers)
		self.dim_in = dim_in
	def forward(self, x):
		return self.fn(x)

class FC2(nn.Module):
	def __init__(self, dim_in, model_config):
		nn.Module.__init__(self)

		hiddens = model_config['hiddens']
		activations = model_config['activations']
		init_weights = model_config['init_weights']
		
		layers = []
		prev_layer_size = dim_in
		
		for size, activation, init_weight in zip(hiddens + [2], activations, init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation))
			prev_layer_size = size

		self.fn = nn.Sequential(*layers)
		self.dim_in = dim_in
	def forward(self, x):
		return self.fn(x)

class FCModel(nn.Module):
	def __init__(self, dim_state, dim_action, model_config):
		nn.Module.__init__(self)

		sample_std = model_config['sample_std']
		fixed_std = model_config['fixed_std']

		policy_hiddens = model_config['policy_hiddens']
		policy_activations = model_config['policy_activations']
		policy_init_weights = model_config['policy_init_weights']

		value_hiddens = model_config['value_hiddens']
		value_activations = model_config['value_activations']
		value_init_weights = model_config['value_init_weights']

		layers = []
		prev_layer_size = dim_state

		for size, activation, init_weight in zip(policy_hiddens + [dim_action], policy_activations, policy_init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation))
			prev_layer_size = size

		layers.append(AppendLogStd(init_log_std=np.log(sample_std), dim=dim_action, fixed_grad = fixed_std))

		self.policy_fn = nn.Sequential(*layers)

		layers = []
		prev_layer_size = dim_state

		for size, activation, init_weight in zip(policy_hiddens + [1], policy_activations, policy_init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation))
			prev_layer_size = size

		self.value_fn = nn.Sequential(*layers)

		self.dim_state = dim_state
		self.dim_action = dim_action

	def forward(self, x):
		logits = self.policy_fn(x)
		value = self.value_fn(x)
		return logits, value