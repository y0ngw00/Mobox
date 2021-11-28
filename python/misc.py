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
				 activation=None,
				 Norm = "None"):
		super(SlimFC, self).__init__()
		layers = []
		linear = nn.Linear(in_size, out_size)
		if Norm == "SpectralNorm":
			linear = nn.utils.spectral_norm(linear)	

		elif Norm == "BatchNorm":
			linear = nn.BatchNorm1d(linear)
		initializer(linear.weight)
		nn.init.constant_(linear.bias, 0.0)
		layers.append(linear)

		


		if activation == "relu":
			layers.append(nn.ReLU())

		if activation == "leaky_relu":
			layers.append(nn.LeakyReLU())
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

class MCMCSampler:
	def __init__(self, epsilon, num_class):
		self.epsilon = epsilon
		self.m0 = 0
		self.target_dist = np.zeros(num_class)
		self.margin = 0.005
		self.num_class = num_class

		
	def update(self,idx, value):

		self.target_dist[idx] = value
		# for each motion class, implement 10 episodes 

	def scaling(self):
		v_max = np.max(self.target_dist)
		v_min = np.min(self.target_dist)
		if v_max - v_min > 1e-6 :
			self.target_dist *= -1
			self.target_dist += v_max 
			self.target_dist /= (v_max - v_min)
			self.target_dist += self.margin	

		else :
			self.target_dist[:] = 1/self.num_class

	def get_distribution():
		return self.target_dist

	def sample(self):
		count =0
		while(1):
			m1 = np.random.randint(self.num_class)
			count += 1
			if self.target_dist[m1] < self.target_dist[self.m0] :
				e = np.random.uniform(low = 0.0, high = 1.0)
				if e < self.epsilon:
					# use this
					self.m0 = m1
					return m1
				else :
					continue

			else :
				# use this
				self.m0 = m1
				return m1

			if count >= 1000:
				#use this 
				self.m0 = m1
				return m1


