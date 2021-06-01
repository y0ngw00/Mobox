import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import copy
import numpy as np

import filter


class FCDiscriminator(object):
	def __init__(self, model, device, config):
		self.model = model

		# self.state_filter = filter.MeanStdFilter(model.dim_in, dataset.mean(axis=0), dataset.std(axis=0))
		self.state_filter = filter.MeanStdRuntimeFilter(model.dim_in)

		self.w_grad = config['w_grad']
		self.w_reg = config['w_reg']
		self.w_decay = config['w_decay']
		self.r_scale = config['r_scale']

		self.grad_clip = config['grad_clip']

		self.optimizer = optim.Adam(self.model.parameters(),lr=config['lr'])

		self.loss = None
		self.device = device
		self.model.to(self.device)

	def __call__(self, ss1):
		if len(ss1.shape) == 1:
			ss1 = ss1.reshape(1, -1)
		ss1_filtered = self.state_filter(ss1, update=False)
		ss1_tensor = self.convert_to_tensor(ss1_filtered)

		d = self.model(ss1_tensor)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))

		return d
	# def __call__(self, ss1, _filter=True):
	# 	ss1 = self.convert_to_ndarray(ss1)
	# 	if _filter:
	# 		ss1_filtered = self.state_filter(ss1, update=False)
	# 	else:
	# 		ss1_filtered = ss1
	# 	ss1_tensor = self.convert_to_tensor(ss1_filtered)

	# 	d = self.model(ss1_tensor)
	# 	d = self.convert_to_ndarray(d)
	# 	d = np.clip(d, -1.0, 1.0)
	# 	d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))

	# 	return d, ss1

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

	def compute_grad_and_line_search(self, s):
		if True:
			return s
		n = s.shape[1]
		n = int(n/2)
		s = s.copy()
		s[:,:n] = s[:,n:]

		s = self.convert_to_tensor(s)
		s.requires_grad = True
		d = self.model(s)
		grad = torch.autograd.grad(outputs=d, 
									inputs=s,
									grad_outputs=torch.ones(d.size()).to(self.device),
									create_graph=False,
									retain_graph=False)[0]
		# alphas = np.array([1.0,0.5,0.25,0.125])
		grad[:,:n].fill_(0.0)
		alphas = [200.0,100.0, 10.0, 1.0, 0.1]
		# s + grad
		s.requires_grad = False
		d_n = []
		for alpha in alphas:
			d_n.append(self.convert_to_ndarray(self.model(s+alpha*grad)))
		d_n.append(self.convert_to_ndarray(d))
		d_n = np.array(d_n)
		alphas.append(0.0)
		alphas = np.array(alphas)
		d_n_max = np.argmax(d_n, axis=0)
		
		s1 = self.convert_to_ndarray(s + self.convert_to_tensor(alphas[d_n_max].reshape(-1,1))*grad)
		s = self.convert_to_ndarray(s)
		s[:,n:] = s1[:,n:]
		return s
	def compute_loss(self, s_expert, s_expert2, s_agent):
		d_expert = self.model(s_expert)
		d_agent  = self.model(s_agent)
		loss_pos = 0.5 * torch.mean(torch.pow(d_expert - 1.0, 2.0))
		loss_neg = 0.5 * torch.mean(torch.pow(d_agent  + 1.0, 2.0))
		''' Compute Accuracy'''
		self.expert_accuracy = torch.sum(d_expert)
		self.agent_accuracy = torch.sum(d_agent)

		self.loss = 0.5 * (loss_pos + loss_neg)

		if self.w_decay>0:
			for i in range(len(self.model.fn)):
				v = self.model.fn[i].model[0].weight
				self.loss += 0.5* self.w_decay * torch.sum(v**2)


		if self.w_reg>0:
			v = self.model.fn[2].model[0].weight
			self.loss += 0.5* self.w_reg * torch.sum(v**2)

		if self.w_grad>0:
			batch_size = s_expert.size()[0]
			s_expert2.requires_grad = True
			d_expert2 = self.model(s_expert2)
			
			grad = torch.autograd.grad(outputs=d_expert2, 
										inputs=s_expert2,
										grad_outputs=torch.ones(d_expert2.size()).to(self.device),
										create_graph=True,
										retain_graph=True)[0]
			
			self.grad_loss = 0.5 * self.w_grad * torch.mean(torch.sum(torch.pow(grad, 2.0), axis=-1))
			self.loss += self.grad_loss
		else:
			self.grad_loss = self.convert_to_tensor(np.array(0.0))

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

	def compute_reward(self, ss1):
		ss1_filtered = self.state_filter(ss1)
		ss1 = self.convert_to_tensor(ss1_filtered)

		d = self.model(ss1)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))
		return d

'''Below function do not use when training'''
import model
import importlib.util

def build_discriminator(dim_state, state_experts, config):
	discriminator_model = model.FC(dim_state, config['discriminator_model'])
	return FCDiscriminator(discriminator_model, torch.device("cpu"), config['discriminator'])

def load_discriminator(discriminator, checkpoint):
	state = torch.load(checkpoint)
	state = state['discriminator_state_dict']
	discriminator.load_state_dict(state)

def load_discriminator_lb(discriminator, checkpoint):
	state = torch.load(checkpoint)
	state = state['discriminator_lb_state_dict']
	discriminator.load_state_dict(state)

def load_discriminator_ub(discriminator, checkpoint):
	state = torch.load(checkpoint)
	state = state['discriminator_ub_state_dict']
	discriminator.load_state_dict(state)