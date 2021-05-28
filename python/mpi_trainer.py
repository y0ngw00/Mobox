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
		if is_root_proc():
			self.create_summary_writer(path)
			self.create_policy(config['model'], config['policy'])
			self.create_disc(config['discriminator_model'], config['discriminator'])
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

		if is_root_proc():
			self.states_expert = self.env.get_states_AMP_expert()
			self.state_dict = {}
			self.state_dict['elapsed_time'] = 0.0
			self.state_dict['num_iterations_so_far'] = 0
			self.state_dict['num_samples_so_far'] = 0
			self.enable_goal = self.env.is_enable_goal()

	def create_summary_writer(self, path):
		self.writer = SummaryWriter(path)

	def create_policy(self, model_config, policy_config):
		p_model = model.FCModel(self.env.get_dim_state(), self.env.get_dim_action(), model_config)
		self.policy = ppo.FCPolicy(p_model, policy_config)

	def create_disc(self, model_config, disc_config):
		d_model = model.FC(self.env.get_dim_state_AMP(), model_config)
		self.disc = discriminator.FCDiscriminator(d_model, disc_config)

	def step(self):
		if is_root_proc():
			self._tic()
		self.generate_samples()
		valid_samples = self.gather_samples()
		if is_root_proc():
			if valid_samples is False:
				return 
			self.compute_TD_GAE()
			self.concat_samples()
			self.optimize()
			self.print_log(self.writer)
			self.save(self.path)

	def _tic(self):
		self.tic = time.time()

	def _toc(self):
		t = 0.0
		t = time.time() - self.tic
		self.state_dict['elapsed_time'] += t
		# self.tic = None
		return t

	def sample_states_expert(self, n):
		m = len(self.states_expert)
		return self.states_expert[np.random.randint(0, m, n)]

	def generate_samples(self):
		if is_root_proc():
			self.log = {}
			self.log['mean_episode_len'] = []
			self.log['mean_episode_reward'] = []
			self.log['mean_episode_reward_goal'] = []
			self.log['num_samples'] = 0
			self.log['episode_lens'] = []

		
		for j in range(self.sample_epoch):
			self.states = gather(self.state)
			if is_root_proc():
				self.states = np.vstack(self.states)
				s, a, log_prob, vf_pred = self.policy(self.states)
				s = s.astype(np.float32)
			else:
				s = None
				a = None
				log_prob = None
				vf_pred = None
			s = scatter(s)
			a = scatter(a)
			log_prob = scatter(log_prob)
			vf_pred = scatter(vf_pred)

			self.env.step(a)
			ss1 = self.env.get_state_AMP()
			eoe = self.env.inspect_end_of_episode()
			rg = self.env.get_reward_goal()
			self.state = self.env.get_state()

			self.episode_buffers[-1].append(Sample(s, a, rg, ss1, vf_pred, log_prob))

			if eoe:
				if len(self.episode_buffers[-1]) != 0:
					self.episode_buffers.append([])
				self.env.reset()
		
	def compute_TD_GAE(self):
		for i in range(len(self.episodes)):
			td_gae = self.policy.compute_ppo_td_gae(self.episodes[i])

			for key, item in td_gae.items():
				self.episodes[i][key] = item

	def gather_samples(self):
		episodes = gather(self.episode_buffers[:-1])
		self.episode_buffers = self.episode_buffers[-1:]

		if is_root_proc():
			self.episodes = []
			for epis in episodes:
				for epi in epis:
					self.episodes.append(epi)
			for i, epi in enumerate(self.episodes):

				s, a, rg, ss1, v, l = Sample(*zip(*epi))
				ss1 = np.vstack(ss1)

				r, ss1 = self.disc(ss1)

				rg = np.array(rg)
				if self.enable_goal:
					r = r*rg
				epi_as_array = {}
				epi_as_array['STATES'] = np.vstack(s)
				epi_as_array['ACTIONS'] = np.vstack(a)
				epi_as_array['STATES_AGENT'] = ss1
				epi_as_array['STATES_EXPERT'] = self.sample_states_expert(len(ss1))
				epi_as_array['REWARDS'] = r.reshape(-1)
				epi_as_array['VF_PREDS'] = np.vstack(v).reshape(-1)
				epi_as_array['LOG_PROBS'] = np.vstack(l).reshape(-1)
				n = len(a)

				self.log['episode_lens'].append(n)
				self.log['mean_episode_reward'].append(epi_as_array['REWARDS'].mean())
				self.log['mean_episode_reward_goal'].append(rg.mean())
				self.log['num_samples'] += n

				self.episodes[i] = epi_as_array
			self.state_dict['num_iterations_so_far'] += 1
			if self.log['num_samples'] == 0:
				self.log['mean_episode_len'] = 0.0
				self.log['mean_episode_reward'] = 0.0
				self.log['mean_episode_reward_goal'] = 0.0
				return False 
			else:
				self.log['mean_episode_len'] = np.mean(self.log['episode_lens'])
				self.log['mean_episode_reward'] = np.mean(self.log['mean_episode_reward'])
				self.log['mean_episode_reward_goal']= np.mean(self.log['mean_episode_reward_goal'])

				self.state_dict['num_samples_so_far'] += self.log['num_samples']
				return True
		else:
			return False

	def concat_samples(self):
		self.samples = {}
		for key, item in self.episodes[0].items():
			self.samples[key] = []

		for epi in self.episodes:
			for key, item in epi.items():
				self.samples[key].append(item)

		for key in self.samples.keys():
			self.samples[key] = np.concatenate(self.samples[key])

		self.samples['ADVANTAGES'] = (self.samples['ADVANTAGES'] - self.samples['ADVANTAGES'].mean())/(1e-4 + self.samples['ADVANTAGES'].std())

	def shuffle_samples(self):
		permutation = np.random.permutation(self.log['num_samples'])

		for key, item in self.samples.items():
			self.samples[key] = item[permutation]
	
	def optimize(self):
		if self.log['num_samples'] == 0:
			self.log['std'] = np.mean(np.exp(self.policy.model.policy_fn[-1].log_std.cpu().detach().numpy()))
			self.log['disc_loss'] = 0.0
			self.log['disc_grad_loss'] = 0.0
			self.log['expert_accuracy'] = 0.0
			self.log['agent_accuracy'] = 0.0
			t = self._toc()
			self.log['t'] = t

			return

		''' Policy '''
		state_filtered = self.policy.state_filter(self.samples['STATES'], update=False)
		self.policy.state_filter(self.samples['STATES'])
		self.samples['STATES'] = state_filtered


		minibatches = []

		cursor = 0
		while cursor < self.log['num_samples']:
			minibatches.append((cursor, cursor + self.sgd_minibatch_size))
			cursor += self.sgd_minibatch_size
		
		for _ in range(self.num_sgd_iter):
			self.shuffle_samples()
			np.random.shuffle(minibatches)
			for minibatch in minibatches:
				states = self.samples['STATES'][minibatch[0]:minibatch[1]]
				actions = self.samples['ACTIONS'][minibatch[0]:minibatch[1]]
				vf_preds = self.samples['VF_PREDS'][minibatch[0]:minibatch[1]]
				log_probs = self.samples['LOG_PROBS'][minibatch[0]:minibatch[1]]
				advantages = self.samples['ADVANTAGES'][minibatch[0]:minibatch[1]]
				value_targets = self.samples['VALUE_TARGETS'][minibatch[0]:minibatch[1]]

				self.policy.compute_loss(states, actions, vf_preds, log_probs, advantages, value_targets)
				self.policy.backward_and_apply_gradients()

		self.log['std'] = np.mean(np.exp(self.policy.model.policy_fn[-1].log_std.cpu().detach().numpy()))
		''' Discriminator '''
		minibatches = []

		cursor = 0
		while cursor < self.log['num_samples']:
			minibatches.append((cursor, cursor + self.disc_sgd_minibatch_size))
			cursor += self.disc_sgd_minibatch_size
		
		state_expert_filtered = self.disc.state_filter(self.samples['STATES_EXPERT'], update=False)
		save_sample_agents = self.samples['STATES_AGENT'].copy()
		state_agent_filtered = self.disc.state_filter(self.samples['STATES_AGENT'], update=False)
		self.disc.state_filter(np.vstack([self.samples['STATES_EXPERT'], self.samples['STATES_AGENT']]))
		self.samples['STATES_EXPERT'] = state_expert_filtered
		self.samples['STATES_AGENT'] = state_agent_filtered

		self.log['disc_loss'] = 0.0
		self.log['disc_grad_loss'] = 0.0
		for _ in range(self.num_disc_sgd_iter):
			self.shuffle_samples()
			np.random.shuffle(minibatches)
			for minibatch in minibatches:
				states_expert = self.samples['STATES_EXPERT'][minibatch[0]:minibatch[1]]
				states_agent = self.samples['STATES_AGENT'][minibatch[0]:minibatch[1]]
				
				self.disc.compute_loss(states_expert, states_agent)
				self.log['disc_loss'] += self.disc.loss.cpu().detach().numpy()
				self.log['disc_grad_loss'] += self.disc.grad_loss.cpu().detach().numpy()
				self.disc.backward_and_apply_gradients()
		self.samples['STATES_AGENT'] = save_sample_agents
		self.log['disc_loss'] = self.log['disc_loss']/self.log['num_samples']
		self.log['disc_grad_loss'] = self.log['disc_grad_loss']/self.log['num_samples']

		self.log['expert_accuracy'] = self.disc.expert_accuracy
		self.log['agent_accuracy'] = self.disc.agent_accuracy
		t = self._toc()
		self.log['t'] = t

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