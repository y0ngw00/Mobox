import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import scipy.signal

import filter

cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def discount(x, gamma):
	return scipy.signal.lfilter([1],[1, -gamma], x[::-1], axis=0)[::-1]

class FCPolicy(object):
	def __init__(self, model, config):
		self.model = model

		self.state_filter = filter.MeanStdFilter(shape=model.dim_state)
		self.distribution = torch.distributions.normal.Normal
		self.gamma = config['gamma']
		self.lb = config['lb']

		self.policy_clip = config['policy_clip']
		self.value_clip = config['value_clip']
		self.grad_clip = config['grad_clip']
		self.w_kl = config['kl'] #ToDo
		self.w_entropy = config['entropy']
		self.optimizer = optim.Adam(self.model.parameters(),lr=config['lr'])

		self.loss = None
		
		self.device = (torch.device("cuda") if cuda else torch.device("cpu"))
		if cuda:
			self.model.cuda()

	def __call__(self, states):
		states_filtered = self.state_filter(states)
		states = self.convert_to_tensor(states_filtered)

		logits, vf_pred = self.model(states)
		
		mean, log_std = torch.chunk(logits, 2, dim = 1)
		action_dists = self.distribution(mean, torch.exp(log_std))
		actions = action_dists.sample()

		logprobs = action_dists.log_prob(actions).sum(-1)

		actions = self.convert_to_ndarray(actions)
		logprobs = self.convert_to_ndarray(logprobs)
		vf_preds = self.convert_to_ndarray(vf_pred)

		return states_filtered, actions, logprobs, vf_preds

	def convert_to_tensor(self, arr, use_cuda=True):
		if torch.is_tensor(arr):
			return arr.to(self.device)
		tensor = torch.from_numpy(np.asarray(arr))
		if tensor.dtype == torch.double:
			tensor = tensor.float()
		if use_cuda:
			return tensor.to(self.device)
		return tensor

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
		# curr_action_dist = 
		states = self.convert_to_tensor(states)
		actions = self.convert_to_tensor(actions)
		vf_preds = self.convert_to_tensor(vf_preds)
		log_probs = self.convert_to_tensor(log_probs)
		advantages = self.convert_to_tensor(advantages)
		value_targets = self.convert_to_tensor(value_targets)
		
		logits, curr_vf_pred = self.model(states)
		
		mean, log_std = torch.chunk(logits, 2, dim = 1)
		curr_action_dist = self.distribution(mean, torch.exp(log_std))

		logp_ratio = torch.exp(
			curr_action_dist.log_prob(actions).sum(-1) - log_probs)

		surrogate_loss = torch.min(
			advantages * logp_ratio,
			advantages * torch.clamp(logp_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip))

		entropy_loss = self.w_entropy * curr_action_dist.entropy().sum(-1)
		# print(entropy_loss)
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

	'''Below function do not use when training'''
	def load_from_path(self, path):
		state = torch.load(path)
		state = state['policy_state_dict']
		self.model.load_state_dict(state['model'])
		self.state_filter.load_state_dict(state['state_filter'])

	def compute_action(self, state, explore):
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

class ElementWiseFCPolicy(object):
	def __init__(self, model, config):
		self.model = model

		self.state_filter0 = filter.MeanStdFilter(shape=model.dim_state0)
		self.state_filter1 = filter.MeanStdFilter(shape=model.dim_state1)

		self.distribution = torch.distributions.normal.Normal
		self.gamma = config['gamma']
		self.lb = config['lb']

		self.policy_clip = config['policy_clip']
		self.value_clip = config['value_clip']
		self.grad_clip = config['grad_clip']
		self.w_kl = config['kl'] #ToDo
		self.w_entropy = config['entropy']
		self.optimizer = optim.Adam(self.model.parameters(),lr=config['lr'])

		self.loss = None
		
		self.device = (torch.device("cuda") if cuda else torch.device("cpu"))
		if cuda:
			self.model.cuda()

	def __call__(self, states):

		s0 = states[0]
		s1 = states[1]

		len_of_s = []
		off_of_s = []
		o = 0
		for s in s1:
			o += len(s)
			off_of_s.append(o)
			len_of_s.append(len(s))
		off_of_s = off_of_s[:-1]
		n = np.sum(len_of_s)
		m = len(s1)
		if n == 0:
			actions = []
			for i in range(m):
				actions.append(np.zeros(0))
			return None, None, actions, None, None

		s1 = np.vstack(s1).astype(np.float32)
		s0_filtered = self.state_filter0(s0)

		s1_filtered = self.state_filter1(s1)
		s0_filtered = s0_filtered.repeat(len_of_s, axis=0)

		s0 = self.convert_to_tensor(s0_filtered)
		s1 = self.convert_to_tensor(s1_filtered)

		logits, vf_preds = self.model(s0, s1, len_of_s)

		means, log_stds = torch.chunk(logits , 2, dim = 1)
		action_dists = self.distribution(means, torch.exp(log_stds))
		actions = action_dists.sample()

		log_probs = action_dists.log_prob(actions)
		len_of_s = np.array(len_of_s)
		
		s0_filtered = np.split(self.convert_to_ndarray(s0_filtered),off_of_s)
		s1_filtered = np.split(self.convert_to_ndarray(s1_filtered),off_of_s)
		vf_preds = np.split(self.convert_to_ndarray(vf_preds),off_of_s)
		actions = np.split(self.convert_to_ndarray(actions),off_of_s)
		log_probs = np.split(self.convert_to_ndarray(log_probs),off_of_s)

		for i in range(m):
			if len(s0_filtered[i]) > 0:
				s0_filtered[i] = s0_filtered[i][0]
				vf_preds[i] = vf_preds[i].mean()
			else:
				vf_preds[i] = 0.0
			actions[i] = actions[i].reshape(-1)
			log_probs[i] = log_probs[i].reshape(-1)
		return s0_filtered, s1_filtered, actions, log_probs, vf_preds

	def convert_to_tensor(self, arr, use_cuda=True):
		if torch.is_tensor(arr):
			return arr.to(self.device)
		tensor = torch.from_numpy(np.asarray(arr.astype(np.float32)))
		if tensor.dtype == torch.double:
			tensor = tensor.float()
		if use_cuda:
			return tensor.to(self.device)
		return tensor

	def convert_to_ndarray(self, arr):
		if isinstance(arr, np.ndarray):
			if arr.dtype == np.float64:
				return arr.astype(np.float32)
			return arr
		return arr.cpu().detach().numpy()

	def compute_ppo_td_gae(self, episode):
		ret = {}
		retsize = len(episode['ACTIONS'])
		vpred_t = np.concatenate([episode['VF_PREDS'], np.array([0.0])])
		delta_t = episode['REWARDS'] + self.gamma * vpred_t[1:] - vpred_t[:-1]

		ret['ADVANTAGES'] = discount(delta_t, self.gamma * self.lb)
		ret['VALUE_TARGETS'] = ret['ADVANTAGES'] + episode['VF_PREDS']

		return ret
	def compute_loss(self, states0, states1, actions, vf_preds, log_probs, advantages, value_targets):
		len_of_s1 = []
		off_of_s1 = []
		o = 0
		for s in states1:
			o += len(s)
			off_of_s1.append(o)
			len_of_s1.append(len(s))
		off_of_s1 = off_of_s1[:-1]
		states0 = states0.repeat(len_of_s1, axis=0)
		states1 = np.vstack(states1).astype(np.float32)
		for i in range(len(actions)):
			actions[i] = actions[i].reshape(-1,self.model.dim_action)
			log_probs[i] = log_probs[i].reshape(-1,self.model.dim_action)
		actions = np.vstack(actions).astype(np.float32)
		log_probs = np.vstack(log_probs).astype(np.float32).sum(-1)
		vf_preds = vf_preds.repeat(len_of_s1)
		advantages = advantages.repeat(len_of_s1)
		value_targets = value_targets.repeat(len_of_s1)

		states0 = self.convert_to_tensor(states0)
		states1 = self.convert_to_tensor(states1)
		actions = self.convert_to_tensor(actions)
		vf_preds = self.convert_to_tensor(vf_preds)
		log_probs = self.convert_to_tensor(log_probs)
		advantages = self.convert_to_tensor(advantages)
		value_targets = self.convert_to_tensor(value_targets)

		logits, curr_vf_preds = self.model(states0, states1, len_of_s1)
		
		mean, log_std = torch.chunk(logits, 2, dim = 1)
		curr_action_dist = self.distribution(mean, torch.exp(log_std))

		logp_ratio = torch.exp(
			curr_action_dist.log_prob(actions).sum(-1) - log_probs)

		# print(f'{logp_ratio.sum()}')
		surrogate_loss = torch.min(
			advantages * logp_ratio,
			advantages * torch.clamp(logp_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip))

		entropy_loss = self.w_entropy * curr_action_dist.entropy().sum(-1)


		curr_vf_preds = curr_vf_preds.reshape(-1)
		vf_loss1 = torch.pow(curr_vf_preds - value_targets , 2.0)
		vf_clipped = vf_preds + torch.clamp(curr_vf_preds - vf_preds, -self.value_clip, self.value_clip)

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
		state['state_filter0'] = self.state_filter0.state_dict()
		state['state_filter1'] = self.state_filter1.state_dict()

		return state

	def load_state_dict(self, state):
		self.model.load_state_dict(state['model'])
		self.optimizer.load_state_dict(state['optimizer'])
		self.state_filter0.load_state_dict(state['state_filter0'])
		self.state_filter1.load_state_dict(state['state_filter1'])

	'''Below function do not use when training'''
	def load_from_path(self, path):
		state = torch.load(path)
		state = state['policyf_state_dict']
		self.model.load_state_dict(state['model'])
		self.state_filter0.load_state_dict(state['state_filter0'])
		self.state_filter1.load_state_dict(state['state_filter1'])

	def compute_action(self, state0, state1, explore):
		n = len(state1)
		if n == 0:
			return np.zeros(0)
		s0 = state0.reshape(1,-1)
		s1 = state1
		
		# s0_filtered = self.state_filter0(s0)
		# s1_filtered = self.state_filter1(s1)

		s0_filtered = self.state_filter0(s0, update=False)
		s1_filtered = self.state_filter1(s1, update=False)
		s0_filtered = s0_filtered.repeat(n, axis=0)

		s0 = self.convert_to_tensor(s0_filtered)
		s1 = self.convert_to_tensor(s1_filtered)

		with torch.no_grad():
			logit, vf_pred = self.model(s0, s1, None)

		mean, log_std = torch.chunk(logit, 2, dim = 1)
		action_dist = self.distribution(mean, torch.exp(log_std))

		# if True:
		if explore:
			action = action_dist.sample()
		else:
			action = action_dist.loc
		# print(action)
		action = self.convert_to_ndarray(action)  #TODO
		self.vf_pred = self.convert_to_ndarray(vf_pred) #TODO
		return action

	def get_vf_pred(self):
		return self.vf_pred

'''Below function do not use when training'''
import model
import importlib.util

def build_policy(dim_state, dim_action, config_path):
	spec = importlib.util.spec_from_file_location("config", config_path)
	spec_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(spec_module)
	spec = spec_module

	config = spec.config
	
	md = model.FCModel(dim_state, dim_action, config['model'])
	
	policy = FCPolicy(md, config['policy'])

	return policy


def build_policyf(dim_state, dim_statef, dim_action, config_path):
	spec = importlib.util.spec_from_file_location("config", config_path)
	spec_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(spec_module)
	spec = spec_module

	config = spec.config
	
	forcemd = model.ElementWiseFCModel(dim_state, dim_statef, dim_action, config['modelf'])
	
	policyf = ElementWiseFCPolicy(forcemd, config['policyf'])

	return policyf