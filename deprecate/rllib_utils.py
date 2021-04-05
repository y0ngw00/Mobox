import ray.rllib.agents.ppo as ppo
import ray
from ray.tune.registry import register_env
import rllib_model_custom_torch
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
import os.path, time, glob

import numpy as np

import argparse
import importlib.util

import ray_env as env_module
from IPython import embed

def init(config, nc=None, ng=None):
	spec = importlib.util.spec_from_file_location("config", config)
	spec_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(spec_module)
	spec = spec_module
	register_env(spec.config['env'], lambda config: env_module.env_cls(config))

	if nc is None:
		nc = spec.num_cpus
		ng = spec.num_gpus
	ray.init(num_cpus=nc, num_gpus=ng)

	
	return spec

def trainer_from_spec(spec):
	return ppo.PPOTrainer(config = spec.config)
	
	

	# return trainer