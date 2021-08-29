import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import scipy.signal

import filter

import time

from misc import *

class PolicyNN(nn.Module):
	def __init__(self, dim_state, dim_class, dim_action, model_config):
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

def discount(x, gamma):
	return scipy.signal.lfilter([1],[1, -gamma], x[::-1], axis=0)[::-1]

class PPO(object):
	def __init__(self, dim_state,  dim_class, dim_action, device, model_config, policy_config):
		self.model = PolicyNN(dim_state, dim_class,  dim_action, model_config)

		self.state_filter = filter.MeanStdRuntimeFilter(shape=self.model.dim_state)
		self.distribution = torch.distributions.normal.Normal
		self.gamma = policy_config['gamma']
		self.lb = policy_config['lb']

		self.policy_clip = policy_config['policy_clip']
		self.value_clip = policy_config['value_clip']
		self.grad_clip = policy_config['grad_clip']
		self.w_kl = policy_config['kl'] #ToDo
		self.w_entropy = policy_config['entropy']
		self.optimizer = optim.Adam(self.model.parameters(),lr=policy_config['lr'])

		self.loss = None
		
		if isinstance(device, str):
			self.device = torch.device(device)
		else:
			self.device = device
		self.model.to(self.device)

	def __call__(self, state):
		if len(state.shape) == 1:
			state = state.reshape(1, -1)
		state_filtered = self.state_filter(state, update=False)
		state_tensor = self.convert_to_tensor(state_filtered)

		with torch.no_grad():
			logit, vf_pred = self.model(state_tensor)
			mean, log_std = torch.chunk(logit, 2, dim = 1)
			
			action_dist = self.distribution(mean, torch.exp(log_std))
			action = action_dist.sample()
			logprob = action_dist.log_prob(action).sum(-1)

		action = self.convert_to_ndarray(action)
		logprob = self.convert_to_ndarray(logprob)
		vf_pred = self.convert_to_ndarray(vf_pred)

		return action, logprob, vf_pred

	def convert_to_tensor(self, arr):
		if torch.is_tensor(arr):
			return arr.to(self.device)
		tensor = torch.from_numpy(np.asarray(arr))
		if tensor.dtype == torch.double:
			tensor = tensor.float()
		return tensor.to(self.device)

	def convert_to_ndarray(self, arr):
		if isinstance(arr, np.ndarray):
			if arr.dtype == np.float64:
				return arr.astype(np.float32)
			return arr
		return arr.cpu().detach().numpy().squeeze()

	def compute_ppo_td_gae(self, episode):
		ret = {}
		retsize = len(episode['ACTIONS'])
		vpred_t = np.concatenate([episode['VF_PREDS'], np.array([0.0])])
		delta_t = episode['REWARDS'] + self.gamma * vpred_t[1:] - vpred_t[:-1]

		ret['ADVANTAGES'] = discount(delta_t, self.gamma * self.lb)
		ret['VALUE_TARGETS'] = ret['ADVANTAGES'] + episode['VF_PREDS']

		return ret
	def compute_loss(self, states, actions, vf_preds, log_probs, advantages, value_targets):
		logits, curr_vf_pred = self.model(states)
		mean, log_std = torch.chunk(logits, 2, dim = 1)
		curr_action_dist = self.distribution(mean, torch.exp(log_std))

		logp_ratio = torch.exp(
			curr_action_dist.log_prob(actions).sum(-1) - log_probs)

		surrogate_loss = torch.min(
			advantages * logp_ratio,
			advantages * torch.clamp(logp_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip))

		entropy_loss = self.w_entropy * curr_action_dist.entropy().sum(-1)

		curr_vf_pred = curr_vf_pred.reshape(-1)
		vf_loss1 = torch.pow(curr_vf_pred - value_targets , 2.0)
		vf_clipped = vf_preds + torch.clamp(curr_vf_pred - vf_preds, -self.value_clip, self.value_clip)

		vf_loss2 = torch.pow(vf_clipped - value_targets, 2.0)
		vf_loss = torch.max(vf_loss1, vf_loss2)
		self.loss = - torch.mean(surrogate_loss) + torch.mean(vf_loss) - torch.mean(entropy_loss)

	def backward_and_apply_gradients(self):
		self.optimizer.zero_grad()
		self.loss.backward(retain_graph = True)
		for param in self.model.parameters():
			if param.grad is not None:
				param.grad.data.clamp_(-self.grad_clip,self.grad_clip)
		self.optimizer.step()
		self.loss = None

	def state_dict(self):
		state = {}
		state['model'] = self.model.state_dict()
		state['optimizer'] = self.optimizer.state_dict()
		state['state_filter'] = self.state_filter.state_dict()

		return state

	def load_state_dict(self, state):
		self.model.load_state_dict(state['model'])
		self.optimizer.load_state_dict(state['optimizer'])
		self.state_filter.load_state_dict(state['state_filter'])
		# self.model.policy_fn[3].set_value(1e-8)

	'''Below function do not use when training'''

	def compute_action(self, state, explore):
		# print(np.any(np.isnan(state)))
		state = state.reshape(1,-1)
		state_filtered = self.state_filter(state, update=False)
		state = self.convert_to_tensor(state_filtered)
		logit, vf_pred = self.model(state)
		mean, log_std = torch.chunk(logit, 2, dim = 1)
		action_dist = self.distribution(mean, torch.exp(log_std))

		if explore:
			action = action_dist.sample()
		else:
			action = action_dist.loc
		action = self.convert_to_ndarray(action)
		self.vf_pred = self.convert_to_ndarray(vf_pred)
		return action

	def get_vf_pred(self):
		return self.vf_pred

	def compute_grad(self, state):
		state = state.reshape(1,-1)
		state_filtered = self.state_filter(state, update=False)
		state = self.convert_to_tensor(state_filtered)
		state = state.squeeze()
		n = self.model.policy_fn[3].log_std.shape[0]
		state = state.repeat(n, 1)
		state.requires_grad = True
		logit, _ = self.model(state)
		mean, log_std = torch.chunk(logit, 2, dim = 1)
		mean.backward(torch.eye(n).cuda())
		return state.grad.data.cpu().detach().numpy() 
'''Below function do not use when training'''
import importlib.util

def load_config(path):
	spec = importlib.util.spec_from_file_location("config", path)
	spec_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(spec_module)
	spec = spec_module
	return spec.config

def build_policy(dim_state, dim_action, config):
	return PPO(dim_state, dim_action, torch.device("cpu"), config['model'], config['policy'])

def load_policy(policy, checkpoint):
	state = torch.load(checkpoint)
	state = state['policy_state_dict']
	policy.load_state_dict(state)