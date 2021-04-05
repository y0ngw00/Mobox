import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import filter

def xavier_initializer(gain=1.0):
	def initializer(tensor):
		torch.nn.init.xavier_uniform_(tensor,gain=gain)
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
	def __init__(self, init_log_std, dim):
		super().__init__()
		self.log_std = torch.nn.Parameter(
			torch.as_tensor([init_log_std] * dim).type(torch.float32))
		self.register_parameter("log_std", self.log_std)
		# self.log_std.requires_grad = True

	def forward(self, x):
		x = torch.cat([x, self.log_std.unsqueeze(0).repeat([len(x), 1])], axis=-1)
		return x

class FCModel(nn.Module):
	def __init__(self, dim_state, dim_action, model_config):
		nn.Module.__init__(self)

		sample_std = model_config['sample_std']

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

		layers.append(AppendLogStd(init_log_std=np.log(sample_std), dim=dim_action))

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

class ElementWiseFCModel(nn.Module):
	def __init__(self, dim_state0, dim_state1, dim_action, model_config):
		nn.Module.__init__(self)

		sample_std = model_config['sample_std']

		policy0_hiddens = model_config['policy0_hiddens']
		policy0_activations = model_config['policy0_activations']
		policy0_init_weights = model_config['policy0_init_weights']
		policy1_hiddens = model_config['policy1_hiddens']
		policy1_activations = model_config['policy1_activations']
		policy1_init_weights = model_config['policy1_init_weights']

		value0_hiddens = model_config['value0_hiddens']
		value0_activations = model_config['value0_activations']
		value0_init_weights = model_config['value0_init_weights']
		value1_hiddens = model_config['value1_hiddens']
		value1_activations = model_config['value1_activations']
		value1_init_weights = model_config['value1_init_weights']


		'''policy 0'''
		layers = []
		prev_layer_size = dim_state0

		for size, activation, init_weight in zip(policy0_hiddens, policy0_activations, policy0_init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation))
			prev_layer_size = size

		self.policy_fn0 = nn.Sequential(*layers)

		'''policy 1'''
		layers = []
		prev_layer_size += dim_state1

		for size, activation, init_weight in zip(policy1_hiddens+[dim_action], policy1_activations, policy1_init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation))
			prev_layer_size = size

		layers.append(AppendLogStd(init_log_std=np.log(sample_std), dim=dim_action))

		self.policy_fn1 = nn.Sequential(*layers)

		'''value 0'''
		layers = []
		prev_layer_size = dim_state0

		for size, activation, init_weight in zip(value0_hiddens, value0_activations, value0_init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation))
			prev_layer_size = size

		self.value_fn0 = nn.Sequential(*layers)

		'''value 1'''
		layers = []
		prev_layer_size += dim_state1

		for size, activation, init_weight in zip(value1_hiddens+[1], value1_activations, value1_init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation))
			prev_layer_size = size

		self.value_fn1 = nn.Sequential(*layers)

		self.dim_state0 = dim_state0
		self.dim_state1 = dim_state1
		self.dim_action = dim_action

	def forward(self, s0, s1, size_of_sections):
		'''1'''
		p0 = self.policy_fn0(s0)
		v0 = self.value_fn0(s0)

		p1 = self.policy_fn1(torch.cat([p0, s1], dim=1))
		v1 = self.value_fn1(torch.cat([v0, s1], dim=1))
		return p1, v1

		'''2'''
		# s1 = torch.split(s1,size_of_sections)
		# p0 = self.policy_fn0(s0)
		# v0 = self.value_fn0(s0)

		# logits_ret = []
		# values_ret = []
		# for _s1,_p0,_v0 in zip(s1, p0, v0):

		# 	_n = _s1.shape[0]
		# 	_p0 = _p0.repeat([_n, 1])
		# 	_v0 = _v0.repeat([_n, 1])
		# 	_p1 = torch.cat([_p0, _s1], dim=1)
		# 	_v1 = torch.cat([_v0, _s1], dim=1)

		# 	logit = self.policy_fn1(_p1)
		# 	value = self.value_fn1(_v1)

		# 	mean, logstd = torch.chunk(logit, 2, dim = 1)

		# 	logits_ret.append(torch.cat([mean.reshape(-1), logstd.reshape(-1)]))
		# 	values_ret.append(value.sum().reshape(1))

		# return logits_ret, torch.cat(values_ret)

		'''3'''
		# s1 = torch.split(s1,size_of_sections)
		# from IPython import embed; embed();exit()
		# p0 = self.policy_fn0(s0)
		# v0 = self.value_fn0(s0)

		# logits_ret = []
		# values_ret = []
		# for _s1,_p0,_v0 in zip(s1, p0, v0):

		# 	_n = _s1.shape[0]
		# 	_p0 = _p0.repeat([_n, 1])
		# 	_v0 = _v0.repeat([_n, 1])
		# 	_p1 = torch.cat([_p0, _s1], dim=1)
		# 	_v1 = torch.cat([_v0, _s1], dim=1)

		# 	logit = self.policy_fn1(_p1)
		# 	value = self.value_fn1(_v1)

		# 	mean, logstd = torch.chunk(logit, 2, dim = 1)

		# 	logits_ret.append(torch.cat([mean.reshape(-1), logstd.reshape(-1)]))
		# 	values_ret.append(value.sum().reshape(1))

		# return logits_ret, torch.cat(values_ret)