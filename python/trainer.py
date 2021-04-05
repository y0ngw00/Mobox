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

Sample = namedtuple('Sample',('s', 'a', 'r', 'vf_pred', 'log_prob'))

class Trainer(object):
	def __init__(self, envs, policy, config):
		self.envs = envs
		self.policy = policy

		self.num_envs = self.envs.get_num_envs()
		self.sample_size = config['sample_size']
		self.sample_epoch = self.sample_size//self.num_envs

		self.num_sgd_iter = config['num_sgd_iter']
		self.sgd_minibatch_size = config['sgd_minibatch_size']

		self.save_iteration = config['save_iteration']
		self.envs.resets()
		self.episode_buffers = []
		for i in range(2):
			self.episode_buffers.append([])

			for j in range(self.num_envs):
				self.episode_buffers[i].append([])
		self.states = self.envs.get_states()

		self.state_dict = {}
		self.state_dict['elapsed_time'] = 0.0
		self.state_dict['num_iterations_so_far'] = 0
		self.state_dict['num_samples_so_far'] = [0, 0]

		self.kinematic = False
		if self.kinematic:
			self.envs.set_kinematics(self.kinematic)
		self.passive = True
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
		self.log['mean_episode_len'] = [[], []]
		self.log['mean_episode_reward'] = [[], []]
		self.log['num_samples'] = [0, 0]

		''' collect episodes '''
		self.episodes = [[], []]
		for j in range(self.sample_epoch):
			s0, a0, lpbs0, vfs0 = self.policy[0](self.states[0])
			s1, a1, lpbs1, vfs1 = self.policy[1](self.states[1])
			s0 = s0.astype(np.float32)
			s1 = s1.astype(np.float32)

			self.envs.steps(a0, a1)
			rs = self.envs.get_rewards()


			eoes = self.envs.inspect_end_of_episodes()
			sleeps = self.envs.is_sleeps()
			self.states = self.envs.get_states()

			for i in range(self.num_envs):
				self.episode_buffers[0][i].append(Sample(s0[i],a0[i],rs[0][i],vfs0[i],lpbs0[i]))
				if sleeps[i] == False:
					self.episode_buffers[1][i].append(Sample(s1[i],a1[i],rs[1][i],vfs1[i],lpbs1[i]))


			for i in range(self.num_envs):
				if eoes[i]:
					self.episodes[0].append(self.episode_buffers[0][i])
					self.episodes[1].append(self.episode_buffers[1][i])
					self.episode_buffers[0][i] = []
					self.episode_buffers[1][i] = []
					self.envs.reset(i)

				if sleeps[i] and len(self.episode_buffers[1][i]) > 0:
					self.episodes[1].append(self.episode_buffers[1][i])
					self.episode_buffers[1][i] = []

		''' Post Process episodes '''
		for j in range(2):
			for i, epi in enumerate(self.episodes[j]):
				if self.kinematic and j == 0:
					continue
				if self.passive and j == 1:
					continue
				s, a, r, v, l = Sample(*zip(*epi))
				epi_as_array = {}
				epi_as_array['STATES'] = np.vstack(s)
				epi_as_array['ACTIONS'] = np.vstack(a)
				epi_as_array['REWARDS'] = np.vstack(r).reshape(-1).astype(np.float32)
				epi_as_array['VF_PREDS'] = np.vstack(v).reshape(-1)
				epi_as_array['LOG_PROBS'] = np.vstack(l).reshape(-1)

				n = len(a)
				self.log['mean_episode_len'][j].append(n)
				self.log['mean_episode_reward'][j].append(epi_as_array['REWARDS'].sum())
				self.log['num_samples'][j] += n

				self.episodes[j][i] = epi_as_array

			if self.log['num_samples'][j] == 0:
				self.log['mean_episode_len'][j] = 0.0
				self.log['mean_episode_reward'][j] = 0.0
			else:
				self.log['mean_episode_len'][j] = np.mean(self.log['mean_episode_len'][j])
				self.log['mean_episode_reward'][j] = np.mean(self.log['mean_episode_reward'][j])

			self.state_dict['num_samples_so_far'][j] += self.log['num_samples'][j]
		self.state_dict['num_iterations_so_far'] += 1

	def compute_TD_GAE(self):
		for j in range(2):
			if self.log['num_samples'][j] == 0:
				continue
			for i in range(len(self.episodes[j])):
				td_gae = self.policy[j].compute_ppo_td_gae(self.episodes[j][i])

				for key, item in td_gae.items():
					self.episodes[j][i][key] = item

	def concat_samples(self):
		self.samples = [{}, {}]
		for j in range(2):
			if self.log['num_samples'][j] == 0:
				continue

			for key, item in self.episodes[j][0].items():
				self.samples[j][key] = []

			for epi in self.episodes[j]:
				for key, item in epi.items():
					self.samples[j][key].append(item)

			for key in self.samples[j].keys():
				self.samples[j][key] = np.concatenate(self.samples[j][key])

	def standarize_samples(self, key):
		for j in range(2):
			if self.log['num_samples'][j] == 0:
				continue

			self.samples[j][key] = (self.samples[j][key] - self.samples[j][key].mean())/(1e-4 + self.samples[j][key].std())


	def shuffle_samples(self):
		for j in range(2):
			if self.log['num_samples'][j] == 0:
				continue
			permutation = np.random.permutation(self.log['num_samples'][j])
			for key, item in self.samples[j].items():
				self.samples[j][key] = item[permutation]

	def optimize(self):
		minibatches = [[], []]
		for j in range(2):
			cursor = 0
			while cursor < self.log['num_samples'][j]:
				minibatches[j].append((cursor, cursor + self.sgd_minibatch_size[j]))
				cursor += self.sgd_minibatch_size[j]
		for _ in range(self.num_sgd_iter):
			self.shuffle_samples()

			for j in range(2):
				if self.log['num_samples'][j] == 0:
					continue
				np.random.shuffle(minibatches[j])
				for minibatch in minibatches[j]:
					states = self.samples[j]['STATES'][minibatch[0]:minibatch[1]]
					actions = self.samples[j]['ACTIONS'][minibatch[0]:minibatch[1]]
					vf_preds = self.samples[j]['VF_PREDS'][minibatch[0]:minibatch[1]]
					log_probs = self.samples[j]['LOG_PROBS'][minibatch[0]:minibatch[1]]
					advantages = self.samples[j]['ADVANTAGES'][minibatch[0]:minibatch[1]]
					value_targets = self.samples[j]['VALUE_TARGETS'][minibatch[0]:minibatch[1]]

					self.policy[j].compute_loss(states, actions, vf_preds, log_probs, advantages, value_targets)
					self.policy[j].backward_and_apply_gradients()
		self.log['std'] = []
		for j in range(2):
			self.log['std'].append(np.mean(np.exp(self.policy[j].model.policy_fn[-1].log_std.cpu().detach().numpy())))
		t = self._toc()
		self.log['t'] = t

		return self.log

	def print_log(self, writer = None):

		def time_to_hms(t):
			h = int((t)//3600.0)
			m = int((t)//60.0)
			s = int((t))
			m = m - h*60
			s = (t)
			s = s - h*3600 - m*60
			return h,m,s
		log = self.log
		
		h,m,s=time_to_hms(self.state_dict['elapsed_time'])
		end = '\n'
		if self.kinematic or self.passive:
			end = ''
		print('# {}, {}h:{}m:{:.1f}s ({:.1f}s)- '.format(self.state_dict['num_iterations_so_far'],h,m,s, self.log['t']),end=end)
		if self.kinematic is False:
			print('policy0   len : {:.1f}, rew : {:.1f}, std : {:.3f} samples : {:,}'.format(log['mean_episode_len'][0],
																						log['mean_episode_reward'][0],
																						log['std'][0],
																						self.state_dict['num_samples_so_far'][0]))
		if self.passive is False:
			print('policy1   len : {:.1f}, rew : {:.1f}, std : {:.3f} samples : {:,}'.format(log['mean_episode_len'][1],
																						log['mean_episode_reward'][1],
																						log['std'][1],
																						self.state_dict['num_samples_so_far'][1]))
		if writer is not None:
			# print(self.state_dict['num_samples_so_far'][1])
			writer.add_scalar('policy0/iterations',self.state_dict['num_iterations_so_far'],
				self.state_dict['num_samples_so_far'][0])
			writer.add_scalar('policy0/episode_len',log['mean_episode_len'][0],
				self.state_dict['num_samples_so_far'][0])
			writer.add_scalar('policy0/reward_mean',log['mean_episode_reward'][0],
				self.state_dict['num_samples_so_far'][0])
			writer.add_scalar('policy0/std',log['std'][0],
				self.state_dict['num_samples_so_far'][0])

			writer.add_scalar('policy1/iterations',self.state_dict['num_iterations_so_far'],
				self.state_dict['num_samples_so_far'][1])
			writer.add_scalar('policy1/episode_len',log['mean_episode_len'][1],
				self.state_dict['num_samples_so_far'][1])
			writer.add_scalar('policy1/reward_mean',log['mean_episode_reward'][1],
				self.state_dict['num_samples_so_far'][1])
			writer.add_scalar('policy1/std',log['std'][1],
				self.state_dict['num_samples_so_far'][1])
	def save(self, path):
		cond0 = self.state_dict['num_iterations_so_far'] % self.save_iteration[0] == 0
		cond1 = self.state_dict['num_iterations_so_far'] % self.save_iteration[1] == 0

		if cond0 or cond1:
			state = {}
			
			state['policy_state_dict'] = [self.policy[0].state_dict(), self.policy[1].state_dict()]

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
		self.policy[0].load_state_dict(state['policy_state_dict'][0])
		self.policy[1].load_state_dict(state['policy_state_dict'][1])

		for key in self.state_dict.keys():
			self.state_dict[key] = state[key]