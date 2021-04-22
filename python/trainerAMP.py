import time

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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from replay_buffer_rand_storage import ReplayBufferRandStorage
Sample = namedtuple('Sample',('s', 'a', 'rg', 'ss1', 'vf_pred', 'log_prob'))

class TrainerAMP(object):
	def __init__(self, envs, policy, discriminator, config):
		self.envs = envs
		self.policy = policy
		self.discriminator = discriminator

		self.num_envs = self.envs.get_num_envs()
		self.sample_size = config['sample_size']
		self.sample_epoch = self.sample_size//self.num_envs

		self.num_sgd_iter = config['num_sgd_iter']
		self.num_disc_sgd_iter = config['num_disc_sgd_iter']

		self.sgd_minibatch_size = config['sgd_minibatch_size']
		self.disc_sgd_minibatch_size = config['disc_sgd_minibatch_size']

		self.save_iteration = config['save_iteration']
		self.states_expert = self.envs.get_states_AMP_expert()

		disc_buffer_len = config['disc_buffer_len']
		disc_buffer_len = int(np.ceil(disc_buffer_len/self.num_envs))

		self.envs.resets()
		self.episode_buffers = []
		for j in range(self.num_envs):
			self.episode_buffers.append([])

		self.states = self.envs.get_states()

		self.disc_expert_buffer = ReplayBufferRandStorage(disc_buffer_len)
		self.disc_agent_buffer = ReplayBufferRandStorage(disc_buffer_len)
		self.state_dict = {}
		self.state_dict['elapsed_time'] = 0.0
		self.state_dict['num_iterations_so_far'] = 0
		self.state_dict['num_samples_so_far'] = 0
		self.pca = PCA(n_components=2)
	def _tic(self):
		self.tic = time.time()

	def _toc(self):
		t = time.time() - self.tic
		self.state_dict['elapsed_time'] += t
		self.tic = None
		return t

	def generate_samples(self):
		self._tic()
		self.log = {}
		self.log['mean_episode_len'] = []
		self.log['mean_episode_reward'] = []
		self.log['mean_episode_reward_goal'] = []
		self.log['num_samples'] = 0

		self.episodes = []
		for j in range(self.sample_epoch):
			s, a, log_prob, vf_pred = self.policy(self.states)
			s = s.astype(np.float32)

			self.envs.steps(a)
			ss1 = self.envs.get_states_AMP()

			eoes = self.envs.inspect_end_of_episodes()
			rgs = self.envs.get_reward_goals()
			self.states = self.envs.get_states()
			for i in range(self.num_envs):
				self.episode_buffers[i].append(Sample(s[i], a[i], rgs[i], ss1[i], vf_pred[i], log_prob[i]))

			for i in range(self.num_envs):
				if eoes[i]:
					self.episodes.append(self.episode_buffers[i])
					self.episode_buffers[i] = []
					self.envs.reset(i)


		for i, epi in enumerate(self.episodes):
			s, a, rg, ss1, v, l = Sample(*zip(*epi))

			ss1 = np.vstack(ss1)
			rg = np.array(rg)
			r, ss1 = self.discriminator(ss1)
			r = 0.5*(r + rg)

			epi_as_array = {}
			epi_as_array['STATES'] = np.vstack(s)
			epi_as_array['ACTIONS'] = np.vstack(a)
			epi_as_array['STATES_AGENT'] = ss1
			epi_as_array['STATES_EXPERT'] = self.sample_states_expert(len(ss1))
			epi_as_array['REWARDS'] = np.vstack(r).reshape(-1)
			epi_as_array['VF_PREDS'] = np.vstack(v).reshape(-1)
			epi_as_array['LOG_PROBS'] = np.vstack(l).reshape(-1)
			n = len(a)
			self.log['mean_episode_len'].append(n)
			self.log['mean_episode_reward'].append(epi_as_array['REWARDS'].mean())
			self.log['mean_episode_reward_goal'].append(rg.mean())
			self.log['num_samples'] += n

			self.episodes[i] = epi_as_array
		
		if self.log['num_samples'] == 0:
			self.log['mean_episode_len'] = 0.0
			self.log['mean_episode_reward'] = 0.0
			self.log['mean_episode_reward_goal'] = 0.0
		else:
			self.log['mean_episode_len'] = np.mean(self.log['mean_episode_len'])
			self.log['mean_episode_reward'] = np.mean(self.log['mean_episode_reward'])
			self.log['mean_episode_reward_goal']= np.mean(self.log['mean_episode_reward_goal'])


			self.state_dict['num_samples_so_far'] += self.log['num_samples']
		self.state_dict['num_iterations_so_far'] += 1
	def compute_TD_GAE(self):
		if self.log['num_samples'] == 0:
			return
		for i in range(len(self.episodes)):
			td_gae = self.policy.compute_ppo_td_gae(self.episodes[i])

			for key, item in td_gae.items():
				self.episodes[i][key] = item
	def concat_samples(self):
		self.samples = {}
		if self.log['num_samples'] == 0:
			return

		for key, item in self.episodes[0].items():
			self.samples[key] = []

		for epi in self.episodes:
			for key, item in epi.items():
				self.samples[key].append(item)

		for key in self.samples.keys():
			self.samples[key] = np.concatenate(self.samples[key])

	def standarize_samples(self, key):
		if self.log['num_samples'] == 0:
			return

		self.samples[key] = (self.samples[key] - self.samples[key].mean())/(1e-4 + self.samples[key].std())

	def shuffle_samples(self):
		if self.log['num_samples'] == 0:
			return
		permutation = np.random.permutation(self.log['num_samples'])
		for key, item in self.samples.items():
			self.samples[key] = item[permutation]
	def sample_states_expert(self, n):
		m = len(self.states_expert)
		return self.states_expert[np.random.randint(0, m, n)]

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

		state_expert_filtered = self.discriminator.state_filter(self.samples['STATES_EXPERT'], update=False)
		state_agent_filtered = self.discriminator.state_filter(self.samples['STATES_AGENT'], update=False)
		self.discriminator.state_filter(np.vstack([self.samples['STATES_EXPERT'], self.samples['STATES_AGENT']]))
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
				
				self.discriminator.compute_loss(states_expert, states_agent)
				self.log['disc_loss'] += self.discriminator.loss.cpu().detach().numpy()
				self.log['disc_grad_loss'] += self.discriminator.grad_loss.cpu().detach().numpy()
				self.discriminator.backward_and_apply_gradients()

		self.log['disc_loss'] = self.log['disc_loss']/self.log['num_samples']
		self.log['disc_grad_loss'] = self.log['disc_grad_loss']/self.log['num_samples']
		

		# embedding_2d = self.pca.fit_transform(np.vstack([self.states_expert, self.samples['STATES_AGENT']]))
		# expert_embed_2d, agent_embed_2d = np.split(embedding_2d, [len(self.states_expert)])
		# self.log['expert_embed_2d'] = expert_embed_2d
		# self.log['agent_embed_2d'] = agent_embed_2d
		self.log['expert_accuracy'] = self.discriminator.expert_accuracy
		self.log['agent_accuracy'] = self.discriminator.agent_accuracy
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
			# if log['num_samples'] != 0:
			# 	figure = plt.figure()
			# 	plt.plot(self.log['expert_embed_2d'][:,0],self.log['expert_embed_2d'][:,1],'ro')
			# 	plt.plot(self.log['agent_embed_2d'][:,0],self.log['agent_embed_2d'][:,1],'bo')
			# 	writer.add_figure("matplotlib/figure", figure)
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
			state['discriminator_state_dict'] = self.discriminator.state_dict()


			for key, item in self.state_dict.items():
				state[key] = item

			if cond0:
				torch.save(state, os.path.join(path,'current.pt'))
				print('save at {}'.format(os.path.join(path,'current.pt')))
			if cond1:
				torch.save(state, os.path.join(path,str(self.state_dict['num_iterations_so_far'])+'.pt'))
				print('save at {}'.format(os.path.join(path,str(self.state_dict['num_iterations_so_far'])+'.pt')))

	def load(self, path):
		state = torch.load(path)
		self.policy.load_state_dict(state['policy_state_dict'])
		self.discriminator.load_state_dict(state['discriminator_state_dict'])

		for key in self.state_dict.keys():
			self.state_dict[key] = state[key]
			
