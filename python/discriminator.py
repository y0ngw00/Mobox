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
		self.layers = []

		self.dim_class = dim_class
		self.dim_in = dim_in 

		prev_layer_size = self.dim_in
		
		for size, activation, init_weight in zip(hiddens, activations[:-1], init_weights[:-1]):
			self.layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation,
				"SpectralNorm"))
			prev_layer_size = size

		self.fn = nn.Sequential(*self.layers)

		self.layers_out =SlimFC(
				prev_layer_size,
				1,
				xavier_initializer(1.0),
				None,
				"SpectralNorm")

		self.fn_out = nn.Sequential(self.layers_out)

		if dim_class > 0:
			self.l_y = nn.utils.spectral_norm(nn.Embedding(dim_class, prev_layer_size))

		
	def forward(self, x, y=None):
		
		h = self.fn(x)
		output = self.fn_out(h)
		if y is not None:
			y = torch.argmax(y, axis=1)
			embed_y = self.l_y(y)*h
			output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)

		return output


class Discriminator(object):
	def __init__(self, dim_state, dim_class, device, model_config, disc_config):
		self.model = DiscriminatorNN(dim_state, dim_class,model_config)
		self.state_filter = filter.MeanStdRuntimeFilter(dim_state)
		self.w_grad = disc_config['w_grad']
		self.w_reg = disc_config['w_reg']
		self.w_decay = disc_config['w_decay']
		self.r_scale = disc_config['r_scale']
		self.loss_type = disc_config['loss']
		self.dim_class = dim_class

		self.grad_clip = disc_config['grad_clip']

		self.optimizer = optim.Adam(self.model.parameters(),lr=disc_config['lr'])

		self.loss = None
		self.device = device
		self.model.to(self.device)

	def __call__(self, ss1 , y):
		if len(ss1.shape) == 1:
			ss1 = ss1.reshape(1, -1)
		ss1_filtered = self.state_filter(ss1, update=False)
		ss1_tensor = self.convert_to_tensor(ss1_filtered)
		y_tensor = self.convert_to_tensor(y, torch.int)

		with torch.no_grad():
			d = self.model(ss1_tensor,y_tensor)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))

		return d


	def convert_to_tensor(self, arr, embed = None):
		if torch.is_tensor(arr):
			return arr.to(self.device)
		tensor = torch.from_numpy(np.asarray(arr))

		if(embed is not None):
			tensor = tensor.int()

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
		y_expert_tensor = self.convert_to_tensor(s_expert[:,-self.dim_class:])
		y_expert2_tensor = self.convert_to_tensor(s_expert2[:,-self.dim_class:])
		y_agent_tensor = self.convert_to_tensor(s_agent[:,-self.dim_class:])

		s_expert_tensor = self.convert_to_tensor(s_expert[:,:-self.dim_class])
		s_expert2_tensor = self.convert_to_tensor(s_expert2[:,:-self.dim_class])
		s_agent_tensor = self.convert_to_tensor(s_agent[:,:-self.dim_class])
			
		d_expert = self.model(s_expert_tensor, y_expert_tensor)
		d_agent  = self.model(s_agent_tensor, y_agent_tensor)
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
				v = self.model.layers[i].model[0].weight
				self.loss += 0.5* self.w_decay * torch.sum(v**2)


		if self.w_reg>0:
			v = self.model.layers_out.model[0].weight
			self.loss += 0.5* self.w_reg * torch.sum(v**2)

		if self.w_grad>0:

			batch_size = s_expert_tensor.size()[0]
			s_expert2_tensor.requires_grad = True
			d_expert2 = self.model(s_expert2_tensor,y_expert2_tensor)
			
			grad = torch.autograd.grad(outputs=d_expert2, 
										inputs=s_expert2_tensor,
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
		ss1_filtered = self.state_filter(ss1[:,:-self.dim_class])
		ss1_tensor = self.convert_to_tensor(ss1_filtered)
		y_tensor = self.convert_to_tensor(ss1[:,-self.dim_class:])

		d = self.model(ss1_tensor, y_tensor)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))
		return d

'''Below function do not use when training'''
import importlib.util


def build_discriminator(dim_state,dim_class,state_experts, config):
	return Discriminator(dim_state-dim_class, dim_class,torch.device(0), config['discriminator_model'], config['discriminator'])


def load_discriminator(discriminator, checkpoint):
	state = torch.load(checkpoint)
	state = state['discriminator_state_dict']
	discriminator.load_state_dict(state)