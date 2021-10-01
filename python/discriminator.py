import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import copy
import numpy as np

import filter

from misc import *

class DiscriminatorNN(nn.Module):
	def __init__(self, dim_in,dim_class, model_config):
		nn.Module.__init__(self)

		hiddens = model_config['hiddens']
		activations = model_config['activations']
		init_weights = model_config['init_weights']
		embedding_length = model_config['embedding_length']
		dim_label_out = model_config['dim_embedding_out']
		self.layers = []

		self.dim_class = dim_class
		self.dim_in = dim_in 

		prev_layer_size = self.dim_in
		
		for size, activation, init_weight in zip(hiddens + [1], activations, init_weights):
			size_modified = size-self.dim_class if size > 1 else size
			self.layers.append(SlimFC(
				prev_layer_size,
				size_modified,
				xavier_initializer(init_weight),
				activation,
				"SpectralNorm"))
			prev_layer_size = size

		self.l0 = self.layers[0]
		self.l1 = self.layers[1]
		self.l2 = self.layers[2]
		
		self.num_layer = 3
		# self.fn = nn.Sequential(*layers)
		
	def forward(self, x):
		add = x[:,-self.dim_class:]
		x1 = self.l0(x)
		x1 = torch.cat((x1,add),1)
		x2 = self.l1(x1)
		x2 = torch.cat((x2,add),1)

		return self.l2(x2)


class Discriminator(object):
	def __init__(self, dim_state, dim_class, device, model_config, disc_config):
		self.model = DiscriminatorNN(dim_state, dim_class,model_config)

		self.state_filter = filter.MeanStdRuntimeFilter(self.model.dim_in)
		self.w_grad = disc_config['w_grad']
		self.w_reg = disc_config['w_reg']
		self.w_decay = disc_config['w_decay']
		self.r_scale = disc_config['r_scale']

		self.grad_clip = disc_config['grad_clip']

		self.optimizer = optim.Adam(self.model.parameters(),lr=disc_config['lr'])

		self.loss = None
		self.device = device
		self.model.to(self.device)

	def __call__(self, ss1):
		if len(ss1.shape) == 1:
			ss1 = ss1.reshape(1, -1)
		ss1_filtered = self.state_filter(ss1, update=False)
		ss1_tensor = self.convert_to_tensor(ss1_filtered)

		with torch.no_grad():
			d = self.model(ss1_tensor)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))

		return d


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
			for i in range(self.model.num_layer):
				v = self.model.layers[i].model[0].weight
				self.loss += 0.5* self.w_decay * torch.sum(v**2)


		if self.w_reg>0:
			v = self.model.layers[2].model[0].weight
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
		if len(ss1.shape) == 1:
			ss1 = ss1.reshape(1, -1)
		ss1_filtered = self.state_filter(ss1)
		ss1_tensor = self.convert_to_tensor(ss1_filtered)

		d = self.model(ss1_tensor)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))
		return d

'''Below function do not use when training'''
import importlib.util

def build_discriminator(dim_state,dim_class,state_experts, config):

	return Discriminator(dim_state, dim_class,torch.device(0), config['discriminator_model'], config['discriminator'])

def load_discriminator(discriminator, checkpoint):
	state = torch.load(checkpoint)
	state = state['discriminator_state_dict']
	discriminator.load_state_dict(state)