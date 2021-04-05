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
Samplef = namedtuple('Samplef',('s0', 's1', 'a', 'r', 'vf_pred', 'log_prob'))
class Trainer(object):
	def __init__(self, envs, policy, policyf, config):
		self.envs = envs
		self.policy = policy
		self.policyf = policyf

		self.num_envs = self.envs.get_num_envs()
		self.sample_size = config['sample_size']
		self.sample_epoch = self.sample_size//self.num_envs

		self.num_sgd_iter = config['num_sgd_iter']
		self.sgd_minibatch_size = config['sgd_minibatch_size']
		self.sgd_minibatch_sizef = config['sgd_minibatch_sizef']

		self.save_iteration = config['save_iteration']
		self.save_iteration2 = config['save_iteration2']

		self.episode_buffers = []
		self.episodef_buffers = []
		for i in range(self.num_envs):
			self.episode_buffers.append([])
			self.episodef_buffers.append([])

		self.states = self.envs.resets()
		self.statefs = self.envs.get_statefs()
		self.sleepfs = self.envs.is_sleepf()
		
		self.samples = {}
		self.samplefs = {}

		self.state_dict = {}
		self.state_dict['elapsed_time'] = 0.0
		self.state_dict['num_samples_so_far'] = 0
		self.state_dict['num_samplefs_so_far'] = 0
		self.state_dict['num_iterations_so_far'] = 0
		
		self.kinematic = False
	def _tic(self):
		self.tic = time.time()

	def _toc(self):
		t = time.time() - self.tic
		self.state_dict['elapsed_time'] += t
		self.tic = None
		
	def generate_samples(self):
		self._tic()

		self.log = {}
		self.log['mean_episode_len'] = []
		self.log['mean_episode_reward'] = []
		self.log['num_samples'] = 0

		self.log['mean_episodef_len'] = []
		self.log['mean_episodef_reward'] = []
		self.log['num_samplefs'] = 0

		self.num_samples = 0
		self.episodes = []
		self.episodefs = []
		for j in range(self.sample_epoch):
			if self.kinematic == False:
				states, actions, logprobs, vf_preds = self.policy(self.states)
			else:
				actions = np.zeros([self.num_envs, self.policy.model.dim_action])
			states0, states1, actionfs, logprobfs, vf_predfs = self.policyf([self.states, self.statefs])
			self.envs.steps(actions, actionfs)
			rewards = self.envs.get_rewards()
			rewardfs = self.envs.get_rewardfs()
			eoes = self.envs.inspect_end_of_episodes()
			for i in range(self.num_envs):
				if self.kinematic == False:
					self.episode_buffers[i].append(Sample(states[i].astype(np.float32),
													actions[i],
													rewards[i],
													vf_preds[i],
													logprobs[i]))
				if self.sleepfs[i] == False:
				# if False:
					self.episodef_buffers[i].append(Samplef(states0[i].astype(np.float32),
															states1[i].astype(np.float32),
															actionfs[i],
															rewardfs[i],
															vf_predfs[i],
															logprobfs[i]))
			self.states = self.envs.get_states()
			for i in range(self.num_envs):
				if eoes[i]:
					if self.kinematic == False:
						self.episodes.append(self.episode_buffers[i])
						self.episode_buffers[i] = []
					self.episodefs.append(self.episodef_buffers[i])
					self.episodef_buffers[i] = []
					self.states[i] = self.envs.reset(i)

				if self.sleepfs[i] and len(self.episodef_buffers[i]) > 0:
					self.episodefs.append(self.episodef_buffers[i])
					self.episodef_buffers[i] = []
			self.statefs = self.envs.get_statefs()
			self.sleepfs = self.envs.is_sleepf()
		for i, epi in enumerate(self.episodes):
			
			
			s,a,r,v,l = Sample(*zip(*epi))
			
			epi_as_array = {}
			epi_as_array['STATES'] = np.vstack(s).astype(np.float32)
			epi_as_array['ACTIONS'] = np.vstack(a)
			epi_as_array['REWARDS'] = np.vstack(r).reshape(-1).astype(np.float32)
			epi_as_array['VF_PREDS'] = np.vstack(v).reshape(-1)
			epi_as_array['LOG_PROBS'] = np.vstack(l).reshape(-1)

			n = len(a)
			self.log['mean_episode_len'].append(n)
			self.log['mean_episode_reward'].append(epi_as_array['REWARDS'].sum())
			self.log['num_samples'] += n

			self.episodes[i] = epi_as_array
		temp_episodef = []
		for i, epi in enumerate(self.episodefs):

			s0, s1, a, r, v, l = Samplef(*zip(*epi))
			if len(a) < 2:
				continue

			epi_as_array = {}
			epi_as_array['STATES'] = [np.vstack(s0), np.array(s1, dtype=np.object)]
			epi_as_array['ACTIONS'] = np.array(a, dtype=np.object)
			epi_as_array['REWARDS'] = np.vstack(r).reshape(-1).astype(np.float32)
			epi_as_array['VF_PREDS'] = np.vstack(v).reshape(-1)
			epi_as_array['LOG_PROBS'] = np.array(l, dtype=np.object)
			n = len(a)
			self.log['mean_episodef_len'].append(n)
			self.log['mean_episodef_reward'].append(epi_as_array['REWARDS'].sum())
			self.log['num_samplefs'] += n
			temp_episodef.append(epi_as_array)
		self.episodefs = temp_episodef

		self.num_samples = self.log['num_samples']
		self.num_samplefs = self.log['num_samplefs']

		if self.num_samples == 0:
			self.log['mean_episode_len'] = 0
			self.log['mean_episode_reward'] = 0
		else:
			self.log['mean_episode_len'] = np.mean(self.log['mean_episode_len'])
			self.log['mean_episode_reward'] = np.mean(self.log['mean_episode_reward'])

		if self.num_samplefs == 0:
			self.log['mean_episodef_len'] = 0
			self.log['mean_episodef_reward'] = 0
		else:
			self.log['mean_episodef_len'] = np.mean(self.log['mean_episodef_len'])
			self.log['mean_episodef_reward'] = np.mean(self.log['mean_episodef_reward'])
		
		self.state_dict['num_samples_so_far'] += self.num_samples
		self.state_dict['num_samplefs_so_far'] += self.num_samplefs
		self.state_dict['num_iterations_so_far'] += 1

	def compute_TD_GAE(self):
		for i in range(len(self.episodes)):
			td_gae = self.policy.compute_ppo_td_gae(self.episodes[i])
			
			for key, item in td_gae.items():
				self.episodes[i][key] = item

		for i in range(len(self.episodefs)):
			td_gae = self.policyf.compute_ppo_td_gae(self.episodefs[i])
			
			for key, item in td_gae.items():
				self.episodefs[i][key] = item

	def concat_samples(self):
		self.samples = {}
		if self.num_samples == 0:
			return
		for key, item in self.episodes[0].items():
			self.samples[key] = []
		
		for episode in self.episodes:
			for key, item in episode.items():
				self.samples[key].append(item)
		
		for key in self.samples.keys():
			self.samples[key] = np.concatenate(self.samples[key])

	def concat_samplefs(self):
		self.samplefs = {}
		if self.num_samplefs == 0:
			return
		for key, item in self.episodefs[0].items():
			self.samplefs[key] = []
		
		for episode in self.episodefs:
			for key, item in episode.items():
				self.samplefs[key].append(item)
		
		key = "STATES"
		item = self.samplefs[key]
		s0s = []
		s1s = []
		for s0, s1 in item:
			s0s.append(s0)
			for _s1 in s1:
				s1s.append(_s1.astype(np.float32))
		self.samplefs[key] = []
		self.samplefs[key].append(np.concatenate(s0s))
		self.samplefs[key].append(np.array(s1s,dtype=np.object))

		key = "ACTIONS"
		item = self.samplefs[key]
		actions = []
		for a in item:
			for _a in a:
				actions.append(_a.astype(np.float32))

		self.samplefs[key] = np.array(actions,dtype=np.object)

		key = "LOG_PROBS"
		item = self.samplefs[key]
		logprobs = []
		for l in item:
			for _l in l:
				logprobs.append(_l.astype(np.float32))

		self.samplefs[key] = np.array(logprobs,dtype=np.object)

		
		
		for key, item in self.samplefs.items():
			if key == "STATES" or key == "ACTIONS" or key =="LOG_PROBS":
				continue
			self.samplefs[key] = np.concatenate(self.samplefs[key])

	def standarize_samples(self, key):
		if self.num_samples != 0:
			self.samples[key] = (self.samples[key] - self.samples[key].mean())/(1e-4 + self.samples[key].std())
		if self.num_samplefs != 0:
			self.samplefs[key] = (self.samplefs[key] - self.samplefs[key].mean())/(1e-4 + self.samplefs[key].std())

	def shuffle_samples(self):
		permutation = np.random.permutation(self.num_samples)
		for key, item in self.samples.items():
			self.samples[key] = item[permutation]

	def shuffle_samplefs(self):
		permutation = np.random.permutation(self.num_samplefs)
		
		
		for key, item in self.samplefs.items():
			if isinstance(item, list):
				self.samplefs[key][0] = item[0][permutation]
				self.samplefs[key][1] = item[1][permutation]
				continue
				
			self.samplefs[key] = item[permutation]

	def optimize(self):
		if self.num_samples == 0:
			self.log['std'] = np.mean(np.exp(self.policy.model.policy_fn[3].log_std.cpu().detach().numpy()))

		cursor = 0
		minibatches = []
		while cursor < self.num_samples:
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
		
		self.log['std'] = np.mean(np.exp(self.policy.model.policy_fn[3].log_std.cpu().detach().numpy()))

	def optimizef(self):
		if self.num_samplefs == 0:
		# if True:
			self._toc()	
			self.log['stdf'] = np.mean(np.exp(self.policyf.model.policy_fn1[3].log_std.cpu().detach().numpy()))
			for key, item in self.state_dict.items():
				self.log[key] = item

			return self.log

		cursor = 0
		minibatches = []
		while cursor < self.num_samplefs:
			minibatches.append((cursor, cursor + self.sgd_minibatch_sizef))
			cursor += self.sgd_minibatch_sizef

		for _ in range(self.num_sgd_iter):
			# self.shuffle_samplefs()
			
			np.random.shuffle(minibatches)
			for minibatch in minibatches:
				states0 = self.samplefs['STATES'][0][minibatch[0]:minibatch[1]]
				states1 = self.samplefs['STATES'][1][minibatch[0]:minibatch[1]]
				actions = self.samplefs['ACTIONS'][minibatch[0]:minibatch[1]]
				vf_preds = self.samplefs['VF_PREDS'][minibatch[0]:minibatch[1]]
				log_probs = self.samplefs['LOG_PROBS'][minibatch[0]:minibatch[1]]
				advantages = self.samplefs['ADVANTAGES'][minibatch[0]:minibatch[1]]
				value_targets = self.samplefs['VALUE_TARGETS'][minibatch[0]:minibatch[1]]

				self.policyf.compute_loss(states0, states1, actions, vf_preds, log_probs, advantages, value_targets)
				self.policyf.backward_and_apply_gradients()
		
		self._toc()
		self.log['stdf'] = np.mean(np.exp(self.policyf.model.policy_fn1[3].log_std.cpu().detach().numpy()))
		for key, item in self.state_dict.items():
			self.log[key] = item

		return self.log

	def save(self, path):
		cond1 = self.state_dict['num_iterations_so_far'] % self.save_iteration == 0
		cond2 = self.state_dict['num_iterations_so_far'] % self.save_iteration2 == 0
		if cond1 or cond2:
			state = {}
			
			state['policy_state_dict'] = self.policy.state_dict()
			state['policyf_state_dict'] = self.policyf.state_dict()

			state['elapsed_time'] = self.state_dict['elapsed_time']
			state['num_samples_so_far'] = self.state_dict['num_samples_so_far']
			state['num_samplefs_so_far'] = self.state_dict['num_samplefs_so_far']
			state['num_iterations_so_far'] = self.state_dict['num_iterations_so_far']

			if cond1:
				torch.save(state, os.path.join(path,'current.pt'))
				print('save at {}'.format(os.path.join(path,'current.pt')))
			if cond2:
				torch.save(state, os.path.join(path,str(self.state_dict['num_iterations_so_far'])+'.pt'))
				print('save at {}'.format(os.path.join(path,str(self.state_dict['num_iterations_so_far'])+'.pt')))

	def load(self, path):
		state = torch.load(path)
		self.policy.load_state_dict(state['policy_state_dict'])
		# self.policyf.load_state_dict(state['policyf_state_dict'])

		self.state_dict['elapsed_time'] = state['elapsed_time']
		self.state_dict['num_samples_so_far'] = state['num_samples_so_far']
		self.state_dict['num_samplefs_so_far'] = state['num_samplefs_so_far']
		self.state_dict['num_iterations_so_far'] = state['num_iterations_so_far']
