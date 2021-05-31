import time
import math
import colorsys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import collections
from collections import namedtuple
from collections import deque

import pickle
import os

import pycomcon
import model
import ppo
import discriminator

from torch.utils.tensorboard import SummaryWriter
from mpi_utils import get_num_procs, get_proc_rank, is_root_proc, broadcast, gather, scatter

Sample = namedtuple('Sample',('s', 'a', 'rg', 'ss1', 'vf_pred', 'log_prob'))

class Trainer(object):
	def __init__(self, env_cls, config, path):
		self.env = env_cls()
		self.path = path

		self.policy = self.create_policy(torch.device("cpu"), config['model'], config['policy'])
		self.disc = self.create_disc(torch.device("cpu"), config['discriminator_model'], config['discriminator'])
		if is_root_proc():
			self.policy_loc = self.create_policy(torch.device("cuda"), config['model'], config['policy'])
			self.disc_loc = self.create_disc(torch.device("cuda"), config['discriminator_model'], config['discriminator'])

		self.num_envs = get_num_procs()

		trainer_config = config['trainer']
		self.sample_size = trainer_config['sample_size']
		self.sample_epoch = self.sample_size//self.num_envs

		self.num_sgd_iter = trainer_config['num_sgd_iter']
		self.num_disc_sgd_iter = trainer_config['num_disc_sgd_iter']

		self.sgd_minibatch_size = trainer_config['sgd_minibatch_size']
		self.disc_sgd_minibatch_size = trainer_config['disc_sgd_minibatch_size']

		self.save_iteration = trainer_config['save_iteration']
		
		self.env.reset()
		self.state = self.env.get_state()

		self.episode_buffers = []
		self.episode_buffers.append([])
		self.states_expert = self.env.get_states_AMP_expert()
		self.enable_goal = self.env.is_enable_goal()

		if is_root_proc():
			self.state_dict = {}
			self.state_dict['elapsed_time'] = 0.0
			self.state_dict['num_iterations_so_far'] = 0
			self.state_dict['num_samples_so_far'] = 0

			self.create_summary_writer(path)

	def create_policy(self, device, model_config, policy_config):
		p_model = model.FCModel(self.env.get_dim_state(), self.env.get_dim_action(), model_config)
		return ppo.FCPolicy(p_model, device, policy_config)

	def create_disc(self, device, model_config, disc_config):
		d_model = model.FC(self.env.get_dim_state_AMP(), model_config)
		return discriminator.FCDiscriminator(d_model, device, disc_config)

	def create_summary_writer(self, path):
		self.writer = SummaryWriter(path)

	def step(self):
		if is_root_proc():
			self._tic()
		self.sync()
		self.generate_episodes()
		self.postprocess_episodes()
		self.gather_episodes()

		if is_root_proc():
			valid_samples = self.concat_samples()
			if valid_samples:
				self.update_filter()
				self.optimize()
			self.print_log(self.writer)
			self.save(self.path)

	def _tic(self):
		self.tic = time.time()

	def _toc(self):
		t = 0.0
		t = time.time() - self.tic
		self.state_dict['elapsed_time'] += t
		return t

	def sample_states_expert(self, n):
		m = len(self.states_expert)
		return self.states_expert[np.random.randint(0, m, n)]

	def sync(self):
		if is_root_proc():
			state_dict = self.policy_loc.state_dict()
			self.policy.load_state_dict(state_dict)
			
			state_dict = self.disc_loc.state_dict()
			self.disc.load_state_dict(state_dict)

		self.policy = broadcast(self.policy)
		self.disc = broadcast(self.disc)

	def generate_episodes(self):
		for j in range(self.sample_epoch):
			a, lp, vf = self.policy(self.state)
			self.env.step(a)

			ss1 = self.env.get_state_AMP()
			eoe = self.env.inspect_end_of_episode()
			rg = self.env.get_reward_goal()
			self.state = self.env.get_state()

			self.episode_buffers[-1].append(Sample(self.state, a, rg, ss1, vf, lp))
			if eoe:
				if len(self.episode_buffers[-1]) != 0:
					self.episode_buffers.append([])
				self.env.reset()
	def postprocess_episodes(self):
		self.episodes = []
		for epi in self.episode_buffers[:-1]:
			s, a, rg, ss1, v, l = Sample(*zip(*epi))
			ss1 = np.vstack(ss1)
			r = self.disc(ss1)

			rg = np.array(rg)

			if self.enable_goal:
				r = 0.5*(r+rg)
			epi_as_array = {}
			epi_as_array['STATES'] = np.vstack(s)
			epi_as_array['ACTIONS'] = np.vstack(a)
			epi_as_array['STATES_AGENT'] = ss1
			epi_as_array['STATES_EXPERT'] = self.sample_states_expert(len(ss1))
			epi_as_array['REWARD_GOALS'] = rg.reshape(-1)
			epi_as_array['REWARDS'] = r.reshape(-1)
			epi_as_array['VF_PREDS'] = np.vstack(v).reshape(-1)
			epi_as_array['LOG_PROBS'] = np.vstack(l).reshape(-1)

			td_gae = self.policy.compute_ppo_td_gae(epi_as_array)

			for key, item in td_gae.items():
				epi_as_array[key] = item

			self.episodes.append(epi_as_array)

		self.episode_buffers = self.episode_buffers[-1:]

	def gather_episodes(self):
		self.episode_list = gather(self.episodes)

	def concat_samples(self):
		self.log = {}
		samples = []
		for epis in self.episode_list:
			for epi in epis:
				samples.append(epi)
		if len(samples) == 0:
			self.log['mean_episode_len'] = 0.0
			self.log['mean_episode_reward'] = 0.0
			self.log['mean_episode_reward_goal'] = 0.0
			return False
		'''vectorize samples'''
		self.samples = {}
		for key, item in samples[0].items():
			self.samples[key] = []

		for sample in samples:
			for key, item in sample.items():
				self.samples[key].append(item)

		for key in self.samples.keys():
			self.samples[key] = np.concatenate(self.samples[key])

		self.samples['ADVANTAGES'] = (self.samples['ADVANTAGES'] - self.samples['ADVANTAGES'].mean())/(1e-4 + self.samples['ADVANTAGES'].std())
		m = len(samples)

		self.log['mean_episode_len'] = len(self.samples['REWARDS'])/m
		self.log['mean_episode_reward'] = np.sum(self.samples['REWARDS'])/m
		self.log['mean_episode_reward_goal'] = np.sum(self.samples['REWARD_GOALS'])/m

		self.state_dict['num_iterations_so_far'] += 1
		self.state_dict['num_samples_so_far'] += len(self.samples['REWARDS'])
		return True

	def update_filter(self):
		self.samples['STATES'] = self.policy_loc.state_filter(self.samples['STATES'])

		n = len(self.samples['STATES_EXPERT'])
		state = self.disc_loc.state_filter(np.vstack([self.samples['STATES_EXPERT'], self.samples['STATES_AGENT']]))

		self.samples['STATES_EXPERT'] = state[:n]
		self.samples['STATES_AGENT'] = state[n:]

	def generate_shuffle_indices(self, batch_size, minibatch_size):
		n = batch_size
		m = minibatch_size
		p = np.random.permutation(n)

		r = m - n%m
		if r>0:
			p = np.hstack([p,np.random.randint(0,n,r)])

		p = p.reshape(-1,m)
		return p

	def optimize(self):
		# XXXXXXXXXXXXXX
		n = len(self.samples['STATES'])
		if n == 0:
			self.log['std'] = np.mean(np.exp(self.policy_loc.model.policy_fn[-1].log_std.cpu().detach().numpy()))
			self.log['disc_loss'] = 0.0
			self.log['disc_grad_loss'] = 0.0
			self.log['expert_accuracy'] = 0.0
			self.log['agent_accuracy'] = 0.0
			self.log['t'] = self._toc()

			return

		''' Policy '''
		self.samples['STATES'] = self.policy_loc.convert_to_tensor(self.samples['STATES'])
		self.samples['ACTIONS'] = self.policy_loc.convert_to_tensor(self.samples['ACTIONS'])
		self.samples['VF_PREDS'] = self.policy_loc.convert_to_tensor(self.samples['VF_PREDS'])
		self.samples['LOG_PROBS'] = self.policy_loc.convert_to_tensor(self.samples['LOG_PROBS'])
		self.samples['ADVANTAGES'] = self.policy_loc.convert_to_tensor(self.samples['ADVANTAGES'])
		self.samples['VALUE_TARGETS'] = self.policy_loc.convert_to_tensor(self.samples['VALUE_TARGETS'])

		for _ in range(self.num_sgd_iter):
			minibatches = self.generate_shuffle_indices(n, self.sgd_minibatch_size)
			for minibatch in minibatches:
				states = self.samples['STATES'][minibatch]
				actions = self.samples['ACTIONS'][minibatch]
				vf_preds = self.samples['VF_PREDS'][minibatch]
				log_probs = self.samples['LOG_PROBS'][minibatch]
				advantages = self.samples['ADVANTAGES'][minibatch]
				value_targets = self.samples['VALUE_TARGETS'][minibatch]

				self.policy_loc.compute_loss(states, actions, vf_preds, log_probs, advantages, value_targets)
				self.policy_loc.backward_and_apply_gradients()

		''' Discriminator '''
		self.samples['STATES_EXPERT'] = self.disc_loc.convert_to_tensor(self.samples['STATES_EXPERT'])
		self.samples['STATES_EXPERT2'] = self.disc_loc.convert_to_tensor(self.samples['STATES_EXPERT'])
		self.samples['STATES_AGENT'] = self.disc_loc.convert_to_tensor(self.samples['STATES_AGENT'])

		disc_loss = 0.0
		disc_grad_loss = 0.0
		expert_accuracy = 0.0
		agent_accuracy = 0.0
		for _ in range(self.num_disc_sgd_iter):
			minibatches = self.generate_shuffle_indices(n, self.sgd_minibatch_size)
			for minibatch in minibatches:
				states_expert = self.samples['STATES_EXPERT'][minibatch]
				states_expert2 = self.samples['STATES_EXPERT2'][minibatch]
				states_agent = self.samples['STATES_AGENT'][minibatch]

				self.disc_loc.compute_loss(states_expert, states_expert2, states_agent)

				disc_loss += self.disc_loc.loss.detach()
				disc_grad_loss += self.disc_loc.grad_loss.detach()
				expert_accuracy += self.disc_loc.expert_accuracy.detach()
				agent_accuracy += self.disc_loc.agent_accuracy.detach()
				self.disc_loc.backward_and_apply_gradients()
		
		'''logging'''
		self.log['std'] = np.mean(np.exp(self.policy_loc.model.policy_fn[-1].log_std.cpu().detach().numpy()))

		self.log['disc_loss'] = disc_loss.cpu().numpy()
		self.log['disc_grad_loss'] = disc_grad_loss.cpu().numpy()
		self.log['expert_accuracy'] = expert_accuracy.cpu().numpy()/n
		self.log['agent_accuracy'] = agent_accuracy.cpu().numpy()/n
		self.log['t'] = self._toc()

		return self.log

	def print_log(self, writer = None):
		def time_to_hms(t):
			h = int((t)//3600.0)
			m = int((t)//60.0)
			s = int((t))
			m = m - h*60
			s = t
			s = s - h*3600 - m*60
			return h,m,s
		log = self.log
		
		h,m,s=time_to_hms(self.state_dict['elapsed_time'])
		end = '\n'
		print('# {}, {}h:{}m:{:.1f}s ({:.1f}s)- '.format(self.state_dict['num_iterations_so_far'],h,m,s, self.log['t']),end=end)
		print('policy   len : {:.1f}, rew : {:.3f}, rew_goal : {:.3f}, std : {:.3f} samples : {:,}'.format(log['mean_episode_len'],
																						log['mean_episode_reward'],
																						log['mean_episode_reward_goal'],
																						log['std'],
																						self.state_dict['num_samples_so_far']))

		print('discriminator loss : {:.3f} grad_loss : {:.3f} acc_expert : {:.3f}, acc_agent : {:.3f}'.format(log['disc_loss'],
																						log['disc_grad_loss'],
																						log['expert_accuracy'],
																						log['agent_accuracy']))
		if writer is not None:
			writer.add_scalar('policy/episode_len',log['mean_episode_len'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('policy/reward_mean',log['mean_episode_reward'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('policy/reward_mean_goal',log['mean_episode_reward_goal'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/loss',log['disc_loss'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/grad_loss',log['disc_grad_loss'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/expert_accuracy',log['expert_accuracy'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/agent_accuracy',log['agent_accuracy'],
				self.state_dict['num_samples_so_far'])

	def save(self, path):
		cond0 = self.state_dict['num_iterations_so_far'] % self.save_iteration[0] == 0
		cond1 = self.state_dict['num_iterations_so_far'] % self.save_iteration[1] == 0

		if cond0 or cond1:
			state = {}
			
			state['policy_state_dict'] = self.policy.state_dict()
			state['discriminator_state_dict'] = self.disc.state_dict()

			for key, item in self.state_dict.items():
				state[key] = item

			if cond0:
				torch.save(state, os.path.join(path,'current.pt'))
				print('save at {}'.format(os.path.join(path,'current.pt')))
			if cond1:
				torch.save(state, os.path.join(path,str(math.floor(self.state_dict['num_samples_so_far']/1e6))+'.pt'))
				print('save at {}'.format(os.path.join(path,str(math.floor(self.state_dict['num_samples_so_far']/1e6))+'.pt')))

	def load(self, path):
		state = torch.load(path)
		self.policy.load_state_dict(state['policy_state_dict'])
		self.discriminator.load_state_dict(state['discriminator_state_dict'])

		for key in self.state_dict.keys():
			self.state_dict[key] = state[key]