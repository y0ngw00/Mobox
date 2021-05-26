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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from replay_buffer_rand_storage import ReplayBufferRandStorage
Sample = namedtuple('Sample',('s', 'a', 'rg', 'ss1', 'vf_pred', 'log_prob'))
Samplelub = namedtuple('Samplelub',('s', 'a', 'rg', 'ss1_lb', 'ss1_ub', 'vf_pred', 'log_prob'))

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

		plt.ion()

		self.figure = plt.figure()
		self.figure.set_figwidth(1920 / self.figure.dpi)
		self.figure.set_figheight(1080 / self.figure.dpi)
		mng = plt.get_current_fig_manager()
		# mng.full_screen_toggle()
		self.envs.resets()
		self.episode_buffers = []
		for j in range(self.num_envs):
			self.episode_buffers.append([])

		self.states = self.envs.get_states()

		self.state_dict = {}
		self.state_dict['elapsed_time'] = 0.0
		self.state_dict['num_iterations_so_far'] = 0
		self.state_dict['num_samples_so_far'] = 0
		self.enable_goal = self.envs.is_enable_goal()
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
		self.log['episode_lens'] = []

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
					if len(self.episode_buffers[i]) != 0:
						self.episodes.append(self.episode_buffers[i])
					self.episode_buffers[i] = []
					self.envs.reset(i)


		for i, epi in enumerate(self.episodes):
			s, a, rg, ss1, v, l = Sample(*zip(*epi))

			ss1 = np.vstack(ss1)

			r, ss1 = self.discriminator(ss1)

			rg = np.array(rg)
			if self.enable_goal:
				# r = 0.5*(r + rg)
				r = r*rg
				# r[np.logical_and(r>0, rg<0)] = -r[np.logical_and(r>0, rg<0)]

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
		
		if self.log['num_samples'] == 0:
			self.log['mean_episode_len'] = 0.0
			self.log['mean_episode_reward'] = 0.0
			self.log['mean_episode_reward_goal'] = 0.0
		else:
			self.log['mean_episode_len'] = np.mean(self.log['episode_lens'])
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

	def sample_states_expert_lb(self, n):
		m = len(self.states_expert_lb)
		return self.states_expert_lb[np.random.randint(0, m, n)]

	def sample_states_expert_ub(self, n):
		m = len(self.states_expert_ub)
		return self.states_expert_ub[np.random.randint(0, m, n)]

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
		cond0 = self.state_dict['num_iterations_so_far'] % self.save_iteration[0] == 0
		cond0 = False
		if cond0:
			state_expert_filtered = self.discriminator.state_filter(self.states_expert[1:], update=False)
			state_agent_filtered = self.discriminator.state_filter(self.samples['STATES_AGENT'], update=False)
			self.log['episode_lens'] = np.array(self.log['episode_lens'])
			for i in range(1,len(self.log['episode_lens'])):
				self.log['episode_lens'][i] += self.log['episode_lens'][i-1]
			self.log['episode_lens'] = self.log['episode_lens'][:-1]
			state_agent_filtered_split = np.split(state_agent_filtered, self.log['episode_lens'])
			self.pca.fit(state_expert_filtered)

			state_filtered = np.vstack([state_expert_filtered, state_agent_filtered])
			embedding_2d = self.pca.transform(state_filtered)

			expert_embed_2d, agent_embed_2d = np.split(embedding_2d, [len(state_expert_filtered)])
			agent_trajectory_embed_2d = []
			for state_agent in state_agent_filtered_split:
				agent_trajectory_embed_2d.append(self.pca.transform(state_agent))

			expert_embed_min = np.min(expert_embed_2d,axis=0)
			expert_embed_max = np.max(expert_embed_2d,axis=0)
			agent_embed_min = np.min(agent_embed_2d,axis=0)
			agent_embed_max = np.max(agent_embed_2d,axis=0)

			embed_min = np.vstack([expert_embed_min,agent_embed_min])
			embed_max = np.vstack([expert_embed_max,agent_embed_max])

			embed_min = np.min(embed_min,axis=0)
			embed_max = np.max(embed_max,axis=0)


			X = embedding_2d[:,0]
			Y = embedding_2d[:,1]
			Z,_ = self.discriminator(state_filtered,False)
			s_new = self.discriminator.compute_grad_and_line_search(state_filtered)
			embedding_2d_new = self.pca.transform(s_new)
			X_new = embedding_2d_new[:,0]
			Y_new = embedding_2d_new[:,1]

			n = state_filtered.shape[1]
			n = int(n/2)
			# print(state_filtered[:,:n]-state_filtered[:,n:])
			# print(state_filtered[0].reshape(2,-1))
			# from IPython import embed; embed();exit()
			# masked_value = 0.2
			# Z[Z<masked_value] = masked_value
			# masked_value = 0.6
			# Z[Z>masked_value] = masked_value

			# slices = 32
			# embed_X = np.arange(embed_min[0],embed_max[0],(embed_max[0] - embed_min[0])/slices)
			# embed_Y = np.arange(embed_min[1],embed_max[1],(embed_max[1] - embed_min[1])/slices)
			# embed_X, embed_Y = np.meshgrid(embed_X, embed_Y)
			# embed_XY = np.vstack([embed_X.reshape(-1),embed_Y.reshape(-1)]).transpose()
			# XY = self.pca.inverse_transform(embed_XY)
			# X = embed_XY[:,0]
			# Y = embed_XY[:,1]
			# Z,_ = self.discriminator(XY,False)
			# if 0:
			ngridx = 100
			ngridy = 100
			xi = np.linspace(embed_min[0], embed_max[0], ngridx)
			yi = np.linspace(embed_min[1], embed_max[1], ngridy)
			triang = tri.Triangulation(X, Y)
			interpolator = tri.LinearTriInterpolator(triang, Z)
			Xi, Yi = np.meshgrid(xi, yi)
			zi = interpolator(Xi, Yi)
			self.figure.clf()
			self.figure.set_figwidth(1280 / self.figure.dpi)
			self.figure.set_figheight(720 / self.figure.dpi)	
			plt.clf()
			plt.contour(xi, yi, zi, levels=15, linewidths=0.5, colors='k')
			cntr = plt.contourf(xi, yi, zi, levels=15, cmap="jet")
			# plt.colorbar(cntr)
			
			# for xx,trj in enumerate(agent_trajectory_embed_2d):
			# 	plt.plot(trj[:,0], trj[:,1],ms=3)
			# 	if xx>1:
			# 		break
			x = expert_embed_2d[:,0]
			y = expert_embed_2d[:,1]
			X = X[len(state_expert_filtered):]
			Y = Y[len(state_expert_filtered):]
			X_new = X_new[len(state_expert_filtered):]
			Y_new = Y_new[len(state_expert_filtered):]
			X = np.split(X, self.log['episode_lens'])
			Y = np.split(Y, self.log['episode_lens'])
			X_new = np.split(X_new, self.log['episode_lens'])
			Y_new = np.split(Y_new, self.log['episode_lens'])
			plt.plot(agent_trajectory_embed_2d[0][:,0], agent_trajectory_embed_2d[0][:,1],'k',dashes=[6, 2],ms=30)
			# for i, p in enumerate(agent_trajectory_embed_2d[0]):
			# 	plt.plot(agent_trajectory_embed_2d[0][i:i+2,0], agent_trajectory_embed_2d[0][i:i+2,1],color=colorsys.hsv_to_rgb(i/len(agent_trajectory_embed_2d[0]),1.0,1.0),ms=20)
			# plt.quiver(X[0],Y[0], (X_new[0]-X[0]), (Y_new[0]-Y[0]), width=0.003,angles='xy', scale_units='xy', scale=1.,color='k')
			plt.plot(expert_embed_2d[:,0], expert_embed_2d[:,1], 'k', ms=5)
			# for xx,trj, in enumerate(agent_trajectory_embed_2d):
				
				# if xx>1:
					# break
			# 

			# plt.plot(X,Y, 'ko', ms=5)
			# plt.plot(X_new,Y_new, 'k+', ms=5)

			# plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1],y[1:]-y[:-1])
			
			
			plt.show()
			plt.pause(0.001)
			# from IPython import embed; embed();exit()
			# from IPython import embed; embed();exit()
			

			# stride = 60




			if 0:
			# if len(agent_trajectory_embed_2d)<3:
				self.figure.clf()
				ax = self.figure.add_subplot(111, projection='3d')
				for i,trj in enumerate(agent_trajectory_embed_2d):
					
					# surf= ax.plot_surface(embed_XY[:,0].reshape(slices,slices),
					# 						embed_XY[:,1].reshape(slices,slices),
					# 						Z.reshape(slices,slices),
					# 						cmap=cm.RdPu,
					# 						antialiased=True)
					# ax.plot(expert_embed_2d[:,0], expert_embed_2d[:,1], 'k', zdir='z', zs=-0.5)
					# for j in range(2):
					# 	# p_data = expert_embed_2d[stride*j:stride*(j+1)]
					# 	# start = np.random.randint(0, len(expert_embed_2d)-stride)
					# 	start = j*stride
					# 	p_data = expert_embed_2d[start:start+stride]
					# 	# p_data = p_data[::10]
					# 	z_dir = np.linspace(0, 1,num=len(p_data))
					# 	ax.plot3D(p_data[:,0], p_data[:,1], z_dir)
					# 	ax.plot3D(p_data[0,0], p_data[0,1], z_dir[0],'k+')
					# 	ax.plot3D(p_data[-1,0], p_data[-1,1], z_dir[-1],'r+')
					# ax.plot(agent_embed_2d[::10,0], agent_embed_2d[::10,1], 'b+', zdir='z', zs=-0.5)
					
					z_dir = np.linspace(0, 1,num=len(trj))
					

					x = trj[:,0]
					y = trj[:,1]
					z = np.linspace(0, 1,num=len(trj))
					ax.quiver(x[:-1], y[:-1], z[:-1], x[1:]-x[:-1], y[1:]-y[:-1], z[1:]-z[:-1],length=0.8,arrow_length_ratio=0.1)
					# ax.plot3D(trj[:,0], trj[:,1], z_dir)
					# ax.plot3D(trj[:,0][0], trj[:,1][0], z_dir[0],'k+')
					# self.figure.colorbar(surf, shrink=0.5, aspect=5)
				ax.view_init(elev=30,azim=70)
				ax.dist=8 
				plt.show()
				plt.savefig(f'fig{i:03d}.png', dpi=300)
				plt.pause(0.001)
				
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
		save_sample_agents = self.samples['STATES_AGENT'].copy()
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
		self.samples['STATES_AGENT'] = save_sample_agents
		self.log['disc_loss'] = self.log['disc_loss']/self.log['num_samples']
		self.log['disc_grad_loss'] = self.log['disc_grad_loss']/self.log['num_samples']

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

			cond0 = self.state_dict['num_iterations_so_far'] % self.save_iteration[0] == 0
			cond0 = False
			if cond0 and self.log['num_samples'] != 0:
				state_expert_filtered = self.discriminator.state_filter(self.states_expert[1:], update=False)
				
				self.log['episode_lens'] = np.array(self.log['episode_lens'])
				for i in range(1,len(self.log['episode_lens'])):
					self.log['episode_lens'][i] += self.log['episode_lens'][i-1]
				self.log['episode_lens'] = self.log['episode_lens'][:-1]
				state_agent_filtered = self.discriminator.state_filter(self.samples['STATES_AGENT'], update=False)
				state_agent_filtered = self.samples['STATES_AGENT'].copy()
				state_agent_filtered_split = np.split(state_agent_filtered, self.log['episode_lens'])
				# state_agent_filtered = state_agent_filtered_split[0]
				self.pca.fit(state_expert_filtered)

				state_filtered = np.vstack([state_expert_filtered, state_agent_filtered])
				embedding_2d = self.pca.transform(state_filtered)

				expert_embed_2d, agent_embed_2d = np.split(embedding_2d, [len(state_expert_filtered)])
				agent_trajectory_embed_2d = []
				for state_agent in state_agent_filtered_split:
					agent_trajectory_embed_2d.append(self.pca.transform(state_agent))

				expert_embed_min = np.min(expert_embed_2d,axis=0)
				expert_embed_max = np.max(expert_embed_2d,axis=0)
				agent_embed_min = np.min(agent_embed_2d,axis=0)
				agent_embed_max = np.max(agent_embed_2d,axis=0)


				embed_min = np.vstack([expert_embed_min,agent_embed_min])
				embed_max = np.vstack([expert_embed_max,agent_embed_max])

				embed_min = np.min(embed_min,axis=0)
				embed_max = np.max(embed_max,axis=0)

				X = embedding_2d[:,0]
				Y = embedding_2d[:,1]
				Z,_ = self.discriminator(state_filtered,False)
				s_new = self.discriminator.compute_grad_and_line_search(state_filtered)
				embedding_2d_new = self.pca.transform(s_new)
				X_new = embedding_2d_new[:,0]
				Y_new = embedding_2d_new[:,1]

				n = state_filtered.shape[1]
				n = int(n/2)

				ngridx = 100
				ngridy = 100
				xi = np.linspace(embed_min[0], embed_max[0], ngridx)
				yi = np.linspace(embed_min[1], embed_max[1], ngridy)
				triang = tri.Triangulation(X, Y)
				interpolator = tri.LinearTriInterpolator(triang, Z)
				Xi, Yi = np.meshgrid(xi, yi)
				zi = interpolator(Xi, Yi)
				
				# self.figure = plt.figure(0)
				# self.figure = plt.gcf()
				# self.figure.clf()
				plt.clf()

				plt.gcf().set_figwidth(1920 / plt.gcf().dpi)
				plt.gcf().set_figheight(1080 / plt.gcf().dpi)
				# self.figure.set_figwidth(320 / self.figure.dpi)
				# self.figure.set_figheight(240 / self.figure.dpi)
				plt.contour(xi, yi, zi, levels=15, linewidths=0.5, colors='k')
				cntr = plt.contourf(xi, yi, zi, levels=15, cmap="jet")

				x = expert_embed_2d[:,0]
				y = expert_embed_2d[:,1]
				X = X[len(state_expert_filtered):]
				Y = Y[len(state_expert_filtered):]
				X_new = X_new[len(state_expert_filtered):]
				Y_new = Y_new[len(state_expert_filtered):]
				X = np.split(X, self.log['episode_lens'])
				Y = np.split(Y, self.log['episode_lens'])
				X_new = np.split(X_new, self.log['episode_lens'])
				Y_new = np.split(Y_new, self.log['episode_lens'])
				plt.plot(agent_trajectory_embed_2d[0][:,0], agent_trajectory_embed_2d[0][:,1],'k',linestyle='--', dashes=(5, 5),ms=10)
				# plt.quiver(X[0],Y[0], (X_new[0]-X[0]), (Y_new[0]-Y[0]), width=0.003,angles='xy', scale_units='xy', scale=1.,color='k')
				plt.plot(expert_embed_2d[:,0], expert_embed_2d[:,1], 'k', ms=20)
				plt.show()
				plt.pause(0.001)

				writer.add_figure('figure/discriminator',plt.gcf(),
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
				torch.save(state, os.path.join(path,str(math.floor(self.state_dict['num_samples_so_far']/1e6))+'.pt'))
				print('save at {}'.format(os.path.join(path,str(math.floor(self.state_dict['num_samples_so_far']/1e6))+'.pt')))

	def load(self, path):
		state = torch.load(path)
		self.policy.load_state_dict(state['policy_state_dict'])
		self.discriminator.load_state_dict(state['discriminator_state_dict'])

		for key in self.state_dict.keys():
			self.state_dict[key] = state[key]
			
