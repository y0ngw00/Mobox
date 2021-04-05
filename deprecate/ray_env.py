import numpy as np

import importlib.util
from os import listdir
from os.path import isfile, join
from collections import deque
from enum import Enum
import itertools
import gym
from gym import spaces
from gym.utils import seeding

import pycomcon

class RayEnv(gym.Env):
	def __init__(self, config):
		super(RayEnv, self).__init__()

		self.env = pycomcon.env()
		
		self.dim_state = self.env.get_dim_state()
		self.dim_action = self.env.get_dim_action()
		self.observation_space = spaces.Box(low=-1e6, high=1e6,shape=[self.dim_state])
		self.action_space = spaces.Box(low=-np.pi, high=np.pi,shape=[self.dim_action])
	def get_force_distribution(self):
		return self.env.get_force_distribution()

	def set_force_distribution(self ,fd):
		return self.env.set_force_distribution(fd)

	def get_ball_distribution(self):
		return self.env.get_ball_distribution()

	def set_ball_distribution(self ,fd):
		return self.env.set_ball_distribution(fd)

	def get_average_force_reward(self):
		return self.env.get_average_force_reward()

	def set_current_force_boundary(self, fb):
		return self.env.set_current_force_boundary(fb)

	def reset(self):
		self.env.reset()
		return self.env.get_state()
	
	def step(self, action):
		self.env.step(action)
		return self.env.get_state(), \
				self.env.get_reward()['r'], \
				self.env.inspect_end_of_episode(),\
				{}
				# self.env.get_info()
				

env_cls = RayEnv