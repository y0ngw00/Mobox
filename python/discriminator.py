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
	def __init__(self, dim_in, dim_class, model_config):
		nn.Module.__init__(self)

		hiddens = model_config['hiddens']
		activations = model_config['activations']
		init_weights = model_config['init_weights']
		embedding_length = model_config['embedding_length']
		dim_label_out = model_config['dim_embedding_out']
		layers = []


		self.dim_in_wolabel = dim_in - dim_class
		self.dim_in = dim_in - dim_class + dim_label_out

		self.label_embedding = nn.Embedding( embedding_length,  dim_label_out)


		prev_layer_size = self.dim_in
		
		for size, activation, init_weight in zip(hiddens + [1], activations, init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation,
				"SpectralNorm"))
			prev_layer_size = size



		self.fn = nn.Sequential(*layers)
		
	def forward(self, x):
		return self.fn(x)


class Discriminator(object):
	def __init__(self, dim_state, dim_class, device, model_config, disc_config):
		self.model = DiscriminatorNN(dim_state, dim_class, model_config)

		self.state_filter = filter.MeanStdRuntimeFilter(self.model.dim_in_wolabel)
		self.w_grad = disc_config['w_grad']
		self.w_reg = disc_config['w_reg']
		self.w_decay = disc_config['w_decay']
		self.r_scale = disc_config['r_scale']
		self.loss_type = disc_config['loss']

		self.grad_clip = disc_config['grad_clip']

		self.optimizer = optim.Adam(self.model.parameters(),lr=disc_config['lr'])
		self.dim_class = dim_class

		self.loss = None
		self.device = device
		self.model.to(self.device)

	def __call__(self, ss1):
		if len(ss1.shape) == 1:
			ss1 = ss1.reshape(1, -1)

		ss1_filtered = np.concatenate((self.state_filter(ss1[:,:-self.dim_class], update=False),ss1[:,-self.dim_class:]),axis=1)
		ss1_tensor = self.convert_to_tensor(ss1_filtered)

		with torch.no_grad():
			label = ss1_tensor[:,-self.dim_class:].long()
			ss1_embed = torch.cat((ss1_tensor[:,:-self.dim_class],self.model.label_embedding(label).float().squeeze(dim=1)),1)

		with torch.no_grad():
			d = self.model(ss1_embed)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))

		return d

	def embedding(self, tensor):
		if tensor.dtype == torch.float:
			tensor = tensor.long().to(self.device)
			out = self.model.label_embedding(tensor)
			return out.float().to(self.device)

		else :
			out = self.model.label_embedding(tensor)
			return out.to(self.device)

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
		if self.loss_type == 'hinge loss':
			zero = torch.Tensor([0]).to(self.device)
			loss_pos = 0.5 * torch.mean(torch.max(zero,-d_expert + 1.0))
			loss_neg = 0.5 * torch.mean(torch.max(zero,d_agent  + 1.0))
		else :
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
			# s_expert2.requires_grad = True
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
		# ss1_filtered = self.state_filter(ss1)
		# ss1 = self.convert_to_tensor(ss1_filtered)

		ss1_filtered = np.concatenate((self.state_filter(ss1[:,:-self.dim_class], update=False),ss1[:,-self.dim_class:]),axis=1)
		ss1_tensor = self.convert_to_tensor(ss1_filtered)

		with torch.no_grad():
			label = ss1_tensor[:,-self.dim_class:].long()
			ss1_embed = torch.cat((ss1_tensor[:,:-self.dim_class],self.model.label_embedding(label).float().squeeze(dim=1)),1)


		d = self.model(ss1_embed)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))
		return d

'''Below function do not use when training'''
import importlib.util

def build_discriminator(dim_state, dim_class,state_experts, config):
	return Discriminator(dim_state, dim_class, torch.device(0), config['discriminator_model'], config['discriminator'])

def load_discriminator(discriminator, checkpoint):
	state = torch.load(checkpoint)
	state = state['discriminator_state_dict']
	discriminator.load_state_dict(state)