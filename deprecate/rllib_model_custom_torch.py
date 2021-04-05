# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import os

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import SlimFC, normc_initializer, AppendBiasLayer
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from IPython import embed
torch, nn = try_import_torch()

import torch.nn.functional as F

logger = logging.getLogger(__name__)

def xavier_initializer(gain=1.0):
	def initializer(tensor):
		torch.nn.init.xavier_uniform_(tensor,gain=gain)

	return initializer

class AppendLogStd(nn.Module):
	def __init__(self, init_log_std, dim):
		super().__init__()
		self.init_log_std = init_log_std

		self.log_std = torch.nn.Parameter(
			torch.as_tensor([init_log_std] * dim))
		self.register_parameter("log_std", self.log_std)
		self.log_std.requires_grad = False
	def forward(self, x):
		if len(x.shape)==3:
			out = torch.cat(
			[x, self.log_std.unsqueeze(0).unsqueeze(0).repeat([len(x), 1, 1])], axis=-1)
		else:
			out = torch.cat(
			[x, self.log_std.unsqueeze(0).repeat([len(x), 1])], axis=-1)
		return out
	
class FCPolicy(TorchModelV2, nn.Module):

	DEFAULT_CONFIG = {
		"sample_std" : 1.0,
		"policy_fn_hiddens" : [256,256],
		"policy_fn_activations" : ["relu","relu", None],
		"policy_fn_init_weights" : [0.1, 0.1, 0.1],
		"value_fn_hiddens" : [256,256],
		"value_fn_activations" : ["relu","relu", None],
		"value_fn_init_weights" : [0.1, 0.1, 0.1]
	}
	def __init__(self, obs_space, action_space, num_outputs, model_config,
				name, **model_kwargs):

		TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
							  model_config, name)

		nn.Module.__init__(self)

		assert num_outputs % 2 == 0, (
			"num_outputs must be divisible by two", num_outputs)
		num_outputs = num_outputs // 2

		custom_model_config = FCPolicy.DEFAULT_CONFIG.copy()
		custom_model_config_by_user = model_config.get("custom_model_config")
		if custom_model_config_by_user:
			custom_model_config.update(custom_model_config_by_user)

		sample_std = custom_model_config.get("sample_std")

		policy_fn_hiddens = custom_model_config.get("policy_fn_hiddens")
		policy_fn_activations = custom_model_config.get("policy_fn_activations")
		policy_fn_init_weights = custom_model_config.get("policy_fn_init_weights")
		
		value_fn_hiddens = custom_model_config.get("value_fn_hiddens")
		value_fn_activations = custom_model_config.get("value_fn_activations")
		value_fn_init_weights = custom_model_config.get("value_fn_init_weights")

		dim_state = int(np.product(obs_space.shape))

		'''Policy Function'''
		layers = []
		prev_layer_size = dim_state

		for i,size in enumerate(policy_fn_hiddens+[num_outputs]):
			layers.append(
				SlimFC(
					in_size = prev_layer_size,
					out_size = size,
					initializer=xavier_initializer(policy_fn_init_weights[i]),
					activation_fn=get_activation_fn(policy_fn_activations[i], framework='torch')))
			prev_layer_size = size
		# print(sample_std)
		layers.append(AppendLogStd(init_log_std=np.log(sample_std), dim=num_outputs))

		self._policy_fn = nn.Sequential(*layers)

		'''Policy Function'''
		layers = []
		prev_layer_size = dim_state

		for i,size in enumerate(value_fn_hiddens+[1]):
			layers.append(
				SlimFC(
					in_size = prev_layer_size,
					out_size = size,
					initializer=xavier_initializer(value_fn_init_weights[i]),
					activation_fn=get_activation_fn(value_fn_activations[i], framework='torch')))
			prev_layer_size = size

		self._value_fn = nn.Sequential(*layers)
		self._cur_value = None
	@override(TorchModelV2)
	def forward(self, input_dict, state, seq_lens):
		obs = input_dict['obs_flat'].float()
		obs = obs.reshape(obs.shape[0], -1)
		logits = self._policy_fn(obs)
		self._cur_value = self._value_fn(obs).squeeze(1)
		
		return logits, state

	@override(TorchModelV2)
	def value_function(self):
		assert self._cur_value is not None, "must call forward() first"
		return self._cur_value
# class FCCNNPolicy(TorchModelV2, nn.Module):
# 	DEFAULT_CONFIG = {
# 		"cnn_dim" : 0,
# 		"cnn_hidden_channels" : [6,16],
# 		"cnn_layers" : 2,
# 		"cnn_init_weights" : 0.1,
# 		"sample_std" : 1.0,
# 		"policy_fn_hiddens" : [256,256],
# 		"policy_fn_activations" : ["relu","relu", None],
# 		"policy_fn_init_weights" : [0.1, 0.1, 0.1],
# 		"value_fn_hiddens" : [256,256],
# 		"value_fn_activations" : ["relu","relu", None],
# 		"value_fn_init_weights" : [0.1, 0.1, 0.1]
# 	}
# 	def __init__(self, obs_space, action_space, num_outputs, model_config,
# 				name, **model_kwargs):

# 		TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
# 							  model_config, name)

# 		nn.Module.__init__(self)

# 		assert num_outputs % 2 == 0, (
# 			"num_outputs must be divisible by two", num_outputs)
# 		num_outputs = num_outputs // 2

# 		custom_model_config = FCPolicy.DEFAULT_CONFIG.copy()
# 		custom_model_config_by_user = model_config.get("custom_model_config")
# 		if custom_model_config_by_user:
# 			custom_model_config.update(custom_model_config_by_user)
# 		assert custom_model_config['cnn_dim'] > 0, ("dim cnn > 0")

# 		sample_std = custom_model_config.get("sample_std")
		
# 		policy_fn_hiddens = custom_model_config.get("policy_fn_hiddens")
# 		policy_fn_activations = custom_model_config.get("policy_fn_activations")
# 		policy_fn_init_weights = custom_model_config.get("policy_fn_init_weights")
		
# 		value_fn_hiddens = custom_model_config.get("value_fn_hiddens")
# 		value_fn_activations = custom_model_config.get("value_fn_activations")
# 		value_fn_init_weights = custom_model_config.get("value_fn_init_weights")

# 		self.cnn_dim = custom_model_config.get('cnn_dim')
		
# 		self.cnn_hidden_channels = custom_model_config.get('cnn_hidden_channels')
# 		cnn_layers = custom_model_config.get("cnn_layers")
# 		cnn_init_weights = custom_model_config.get("cnn_init_weights")

# 		dim_state = int(np.product(obs_space.shape))

# 		'''Policy Function'''
# 		cnn_layers = []
# 		prev_size = 3

# 		for i,size in enumerate(self.cnn_hidden_channels):
# 			cnn_layers.append(nn.Conv1d(prev_size,size,2))
# 			cnn_layers.append(nn.MaxPool1d(2))
# 			prev_size = size
# 		self._policy_cnn_fn = nn.Sequential(*cnn_layers)
# 		test_input = torch.zeros(1,3,prev_layer_size)
# 		cnn_size = self._policy_cnn_fn(test_input).reshape(-1)

# 		layers = []
# 		prev_layer_size = dim_state - self.cnn_dim + cnn_size

# 		for i,size in enumerate(policy_fn_hiddens+[num_outputs]):
# 			layers.append(
# 				SlimFC(
# 					in_size = prev_layer_size,
# 					out_size = size,
# 					initializer=xavier_initializer(policy_fn_init_weights[i]),
# 					activation_fn=get_activation_fn(policy_fn_activations[i], framework='torch')))
# 			prev_layer_size = size

# 		layers.append(AppendLogStd(init_log_std=np.log(sample_std), dim=num_outputs))

# 		self._policy_fn = nn.Sequential(*layers)

# 		'''Policy Function'''
# 		cnn_layers = []
# 		prev_size = 3

# 		for i,size in enumerate(self.cnn_hidden_channels):
# 			cnn_layers.append(nn.Conv1d(prev_size,size,2))
# 			cnn_layers.append(nn.MaxPool1d(2))
# 			prev_size = size
# 		self._value_cnn_fn = nn.Sequential(*cnn_layers)
# 		test_input = torch.zeros(1,3,prev_layer_size)
# 		cnn_size = self._value_cnn_fn(test_input).reshape(-1)

# 		layers = []
# 		prev_layer_size = dim_state - self.cnn_dim + cnn_size

# 		for i,size in enumerate(value_fn_hiddens+[1]):
# 			layers.append(
# 				SlimFC(
# 					in_size = prev_layer_size,
# 					out_size = size,
# 					initializer=xavier_initializer(value_fn_init_weights[i]),
# 					activation_fn=get_activation_fn(value_fn_activations[i], framework='torch')))
# 			prev_layer_size = size

# 		self._value_fn = nn.Sequential(*layers)
# 		self._cur_value = None
# 	@override(TorchModelV2)
# 	def forward(self, input_dict, state, seq_lens):
# 		obs = input_dict['obs_flat'].float()
# 		obs = obs.reshape(obs.shape[0], -1)
# 		obs_cnn, obs_fc = obs.split([self.cnn_dim,obs.shape[1]-self.cnn_dim], dim = 1)

# 		obs_cnn = self._policy_cnn_fn(obs_cnn)

# 		logits = self._policy_fn(obs)
# 		self._cur_value = self._value_fn(obs).squeeze(1)

# 		return logits, state

# 	@override(TorchModelV2)
# 	def value_function(self):
# 		assert self._cur_value is not None, "must call forward() first"
# 		return self._cur_value
class RNNModel(TorchRNN, nn.Module):
	DEFAULT_CONFIG = {
		"rnn_dim" : 0,
		"rnn_hiddens" : 16,
		"rnn_layers" : 2,
		"rnn_init_weights" : 0.1,
		"sample_std" : 1.0,
		"policy_fn_hiddens" : [256,256],
		"policy_fn_activations" : ["relu","relu", None],
		"policy_fn_init_weights" : [0.1, 0.1, 0.1],
		"value_fn_hiddens" : [256,256],
		"value_fn_activations" : ["relu","relu", None],
		"value_fn_init_weights" : [0.1, 0.1, 0.1],

	}

	def __init__(self, obs_space, action_space, num_outputs, model_config, name,**model_kwargs):
		nn.Module.__init__(self)
		TorchRNN.__init__(self,obs_space, action_space, num_outputs, model_config, name)

		assert num_outputs % 2 == 0, (
			"num_outputs must be divisible by two", num_outputs)
		num_outputs = num_outputs // 2

		custom_model_config = RNNModel.DEFAULT_CONFIG.copy()
		custom_model_config_by_user = model_config.get("custom_model_config")
		if custom_model_config_by_user:
			custom_model_config.update(custom_model_config_by_user)
		assert custom_model_config['rnn_dim'] > 0, ("dim rnn > 0")

		sample_std = custom_model_config.get("sample_std")
		
		policy_fn_hiddens = custom_model_config.get("policy_fn_hiddens")
		policy_fn_activations = custom_model_config.get("policy_fn_activations")
		policy_fn_init_weights = custom_model_config.get("policy_fn_init_weights")
		
		value_fn_hiddens = custom_model_config.get("value_fn_hiddens")
		value_fn_activations = custom_model_config.get("value_fn_activations")
		value_fn_init_weights = custom_model_config.get("value_fn_init_weights")

		rnn_dim = custom_model_config['rnn_dim']
		self.rnn_dim = rnn_dim
		rnn_hiddens = custom_model_config['rnn_hiddens']
		self.rnn_hiddens = rnn_hiddens
		rnn_layers = custom_model_config['rnn_layers']
		rnn_init_weights = custom_model_config['rnn_init_weights']
		self.rnn_layers = rnn_layers
		self.rnn_state_size = rnn_hiddens
		dim_state = int(np.product(obs_space.shape))

		'''Policy Function'''
		self._rnn_fn = nn.LSTM(rnn_dim,rnn_hiddens,self.rnn_layers)
		for param in self._rnn_fn.parameters():
			if len(param.shape) == 2:
				nn.init.xavier_uniform_(param,gain = rnn_init_weights)
			else:
				param.data.zero_()

		layers = []
		prev_layer_size = dim_state - rnn_dim + rnn_hiddens

		for i,size in enumerate(policy_fn_hiddens+[num_outputs]):
			layers.append(
				SlimFC(
					in_size = prev_layer_size,
					out_size = size,
					initializer=xavier_initializer(policy_fn_init_weights[i]),
					activation_fn=get_activation_fn(policy_fn_activations[i], framework='torch')))
			prev_layer_size = size

		layers.append(AppendLogStd(init_log_std=np.log(sample_std), dim=num_outputs))
		self._policy_fn = nn.Sequential(*layers)

		'''Value Function'''

		layers = []
		prev_layer_size = dim_state - rnn_dim + rnn_hiddens

		for i,size in enumerate(value_fn_hiddens+[1]):
			layers.append(
				SlimFC(
					in_size = prev_layer_size,
					out_size = size,
					initializer=xavier_initializer(value_fn_init_weights[i]),
					activation_fn=get_activation_fn(value_fn_activations[i], framework='torch')))
			prev_layer_size = size

		self._value_fn = nn.Sequential(*layers)

		# self.obs_size = get_preprocessor(obs_space)(obs_space).size
		# self.fc_size = fc_size
		# self.lstm_state_size = lstm_state_size

		# # Build the Module from fc + LSTM + 2xfc (action + value outs).
		# self.fc1 = nn.Linear(self.obs_size, self.fc_size)
		# self.lstm = nn.LSTM(
		# 	self.fc_size, self.lstm_state_size, batch_first=True)
		# self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
		# self.value_branch = nn.Linear(self.lstm_state_size, 1)
		# # Holds the current "base" output (before logits layer).
		# self._features = None
		self._rnn_features = None
	@override(ModelV2)
	def get_initial_state(self):
		
		h = [
			torch.zeros(self.rnn_layers,self.rnn_state_size),
			torch.zeros(self.rnn_layers,self.rnn_state_size)]
		return h

	@override(ModelV2)
	def value_function(self):
		assert self._rnn_features is not None, "must call forward() first"
		return torch.reshape(self._cur_value, [-1])

	@override(TorchRNN)
	def forward_rnn(self, inputs, state, seq_lens):
		"""Feeds `inputs` (B x T x ..) through the Gru Unit.

		Returns the resulting outputs as a sequence (B x T x ...).
		Values are stored in self._cur_value in simple (B) shape (where B
		contains both the B and T dims!).

		Returns:
			NN Outputs (B x T x ...) as sequence.
			The state batches as a List of two items (c- and h-states).
		"""

		# x = nn.functional.relu(self.fc1(inputs))
		
		rnn_dim = self.rnn_dim
		fc_dim = inputs.shape[-1]-rnn_dim
		rnn_inputs, fc_inputs = inputs.split([rnn_dim,fc_dim],dim=-1)
		self._rnn_features, [h,c] = self._rnn_fn(rnn_inputs,[state[0].squeeze(0).unsqueeze(1),state[1].squeeze(0).unsqueeze(1)])

		fc_inputs = torch.cat([fc_inputs,self._rnn_features],dim=-1)
		action_out = self._policy_fn(fc_inputs)

		self._cur_value = self._value_fn(fc_inputs)
		return action_out, [torch.squeeze(h, 1), torch.squeeze(c, 1)]

# class MultiAgentFCPolicy(TorchModelV2, nn.Module):

# 	DEFAULT_CONFIG = {
# 		"sample_std" : 1.0,
# 		"policy_fn_hiddens" : [256,256],
# 		"policy_fn_activations" : ["relu","relu", None],
# 		"policy_fn_init_weights" : [0.1, 0.1, 0.1],
# 		"value_fn_hiddens" : [256,256],
# 		"value_fn_activations" : ["relu","relu", None],
# 		"value_fn_init_weights" : [0.1, 0.1, 0.1],
# 		"num_agents" : 2,
# 	}
# 	def __init__(self, obs_space, action_space, num_outputs, model_config,
# 				name, **model_kwargs):

# 		TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
# 							  model_config, name)

# 		nn.Module.__init__(self)

# 		assert num_outputs % 2 == 0, (
# 			"num_outputs must be divisible by two", num_outputs)
# 		num_outputs = num_outputs // 2

# 		custom_model_config = FCPolicy.DEFAULT_CONFIG.copy()
# 		custom_model_config_by_user = model_config.get("custom_model_config")
# 		if custom_model_config_by_user:
# 			custom_model_config.update(custom_model_config_by_user)

# 		sample_std = custom_model_config.get("sample_std")
		
# 		policy_fn_hiddens = custom_model_config.get("policy_fn_hiddens")
# 		policy_fn_activations = custom_model_config.get("policy_fn_activations")
# 		policy_fn_init_weights = custom_model_config.get("policy_fn_init_weights")
		
# 		value_fn_hiddens = custom_model_config.get("value_fn_hiddens")
# 		value_fn_activations = custom_model_config.get("value_fn_activations")
# 		value_fn_init_weights = custom_model_config.get("value_fn_init_weights")
# 		self._num_agents = custom_model_config.get("num_agents")

# 		dim_state = int(np.product(obs_space.shape))

# 		'''Policy Function'''
# 		self._policy_fns = []
		
# 		for j in self._num_agents:
# 			layers = []
# 			prev_layer_size = dim_state // self._num_agents

# 			for i,size in enumerate(policy_fn_hiddens+[num_outputs//self._num_agents]):
# 				layers.append(
# 					SlimFC(
# 						in_size = prev_layer_size,
# 						out_size = size,
# 						initializer=xavier_initializer(policy_fn_init_weights[i]),
# 						activation_fn=get_activation_fn(policy_fn_activations[i], framework='torch')))
# 				prev_layer_size = size

# 			self._policy_fns.append(nn.Sequential(*layers))
# 			self.add_module('agent_'+str(i),self._policy_fns[-1])
# 		self._policy_fns.append(AppendLogStd(init_log_std=np.log(sample_std), dim=num_outputs))
# 		self.add_module('appendLogStd',self._policy_fns[-1])
# 		'''Value Function'''
# 		self._value_fns = []
# 		for j in self._num_agents:
# 			layers = []
# 			prev_layer_size = dim_state // self._num_agents

# 			for i,size in enumerate(value_fn_hiddens):
# 				layers.append(
# 					SlimFC(
# 						in_size = prev_layer_size,
# 						out_size = size,
# 						initializer=xavier_initializer(value_fn_init_weights[i]),
# 						activation_fn=get_activation_fn(value_fn_activations[i], framework='torch')))
# 				prev_layer_size = size

# 			self._value_fns.append(nn.Sequential(*layers))
# 			self.add_module('value_'+str(i),self._value_fns[-1])
# 		self._value_fns.append(
# 					SlimFC(in_size=prev_layer_size*self._num_agents,
# 							out_size = 1,
# 							initializer=xavier_initializer(value_fn_init_weights[i]),
# 							activation_fn=get_activation_fn(value_fn_activations[i], framework='torch')))
# 		self._cur_value = None
# 	@override(TorchModelV2)
# 	def forward(self, input_dict, state, seq_lens):
# 		obs = input_dict['obs_flat'].float()
# 		obs = obs.reshape(obs.shape[0], -1)

# 		obs = obs.chunk(self._num_agents,axis=-1)

# 		logits = []
# 		for i in range(self._num_agents):
# 			logits.append(self._policy_fns[i](obs[i]))

# 		logits = torch.cat(logits, axis=-1)
# 		logits = self._policy_fns[-1](logits)

# 		values = []
# 		for i in range(self._num_agents):
# 			values.append(self._value_fns[i](obs[i]))

# 		values = torch.cat(values, axis=-1)
# 		self._cur_value = self._value_fns[-1](values).squeeze(1)

# 		return logits, state

# 	@override(TorchModelV2)
# 	def value_function(self):
# 		assert self._cur_value is not None, "must call forward() first"
# 		return self._cur_value


# class MOEPolicy(TorchModelV2, nn.Module):
# 	def __init__(self, obs_space, action_space, num_outputs, model_config,
# 				 name, **model_kwargs):
# 		TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
# 							  model_config, name)
# 		nn.Module.__init__(self)
# 		''' Load and check configuarations '''

# 		assert num_outputs % 2 == 0, (
# 			"num_outputs must be divisible by two", num_outputs)
# 		num_outputs = num_outputs // 2

		
# 		dim_state = int(np.product(obs_space.shape))
# 		num_experts = 8

# 		hiddens = [256,256]

# 		self._use_multiplicative = False
# 		'''Construct expert fn'''
# 		self._experts = []

# 		for i in range(num_experts):
# 			layers = []
# 			prev_layer_size = dim_state

# 			for size in hiddens:
# 				layers.append(
# 					SlimFC(
# 						in_size = prev_layer_size,
# 						out_size = size,
# 						initializer=normc_initializer(1.0),
# 						activation_fn=nn.ReLU))
# 				prev_layer_size = size

# 			layers.append(SlimFC(
# 				in_size = prev_layer_size,
# 				# out_size = num_outputs*2 if self._use_multiplicative else num_outputs,
# 				out_size = num_outputs,
# 				initializer=normc_initializer(0.01),
# 				activation_fn = None))
# 			self.min_log_std = np.log(0.05)
# 			# torch.nn.init.constant_(layers[-1]._model[0].bias[num_outputs:],self.min_log_std)

# 			self._experts.append(nn.Sequential(*layers))
# 			self.add_module('expert_'+str(i),self._experts[-1])
# 		self._append_log_std = AppendBiasLayer(num_outputs)
# 		'''Construct gate fn'''
# 		layers = []
# 		prev_layer_size = dim_state
# 		for size in hiddens:
# 			layers.append(
# 				SlimFC(
# 					in_size = prev_layer_size,
# 					out_size = size,
# 					initializer=normc_initializer(1.0),
# 					activation_fn=nn.ReLU))
# 			prev_layer_size = size

# 		layers.append(SlimFC(
# 			in_size = prev_layer_size,
# 			out_size = num_experts,
# 			initializer=normc_initializer(0.01),
# 			activation_fn = None))

# 		self._gate_fn = nn.Sequential(*layers)

# 		'''Construct value fn'''
# 		prev_layer_size = dim_state
# 		layers = []
# 		for size in hiddens:
# 			layers.append(
# 				SlimFC(
# 					in_size = prev_layer_size,
# 					out_size = size,
# 					initializer=normc_initializer(1.0),
# 					activation_fn=nn.ReLU))
# 			prev_layer_size = size
# 		layers.append(SlimFC(
# 			in_size = prev_layer_size,
# 			out_size = 1,
# 			initializer=normc_initializer(1.0),
# 			activation_fn = None))

# 		self._value_fn = nn.Sequential(*layers)
# 		self._cur_value = None
# 	@override(TorchModelV2)
# 	def forward(self, input_dict, state, seq_lens):
# 		obs = input_dict["obs_flat"].float()
# 		obs = obs.reshape(obs.shape[0], -1)
# 		if self._use_multiplicative:
# 			w = F.softmax(self._gate_fn(obs), dim = 1).unsqueeze(-1)
# 			x = []
# 			for expert in self._experts:
# 				x.append(expert(obs))
# 			x = torch.stack(x,dim = 1)

# 			std = 1.0 / torch.sum(w, dim = 1)
# 			logits = std * torch.sum(w*x, dim = 1)
# 			# stds = self.min_log_std*torch.ones(x.shape,dtype=x.dtype,device=x.device)
# 			# z = w/stds

# 			# std = 1.0 / torch.sum(w, dim = 1)
# 			# logits = std * torch.sum(z * means, dim = 1)
# 		else:
# 			w = F.softmax(self._gate_fn(obs), dim = 1)
# 			logits = None
# 			for i, expert in enumerate(self._experts):
# 				if logits is None:
# 					logits = w[...,i].unsqueeze(-1)*expert(obs)
# 				else:
# 					logits += w[...,i].unsqueeze(-1)*expert(obs)

# 		logits = self._append_log_std(logits)
# 		self._cur_value = self._value_fn(obs).squeeze(1)

# 		return logits, state

# 	@override(TorchModelV2)
# 	def value_function(self):
# 		assert self._cur_value is not None, "must call forward() first"
# 		return self._cur_value

ModelCatalog.register_custom_model("fc", FCPolicy)
ModelCatalog.register_custom_model("rnn", RNNModel)
# ModelCatalog.register_custom_model("mafc", MultiAgentFCPolicy)
# ModelCatalog.register_custom_model("moe", MOEPolicy)