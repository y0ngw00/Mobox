import argparse
import importlib.util
import datetime

import os
import numpy as np

import ray

from ray import tune
from ray.tune.registry import register_env

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog

import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer as Trainer
import rllib_utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ray_env as env_module
import rllib_model_custom_torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
# from tensorboardX import SummaryWriter




from IPython import embed
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', required=True, type=str)
	parser.add_argument("--checkpoint", type=str, default=None)
	args = parser.parse_args()

	spec = rllib_utils.init(args.config)
	trainer = rllib_utils.trainer_from_spec(spec)
	
	if args.checkpoint is not None:
		trainer.restore(args.checkpoint)
	# else:
	# 	pass # todo : remove previous learning curve

	
	dyear = datetime.datetime.now().year
	dmonth = datetime.datetime.now().month
	dday = datetime.datetime.now().day
	dhour = datetime.datetime.now().hour
	dminute = datetime.datetime.now().minute
	dsecond = datetime.datetime.now().second
	date_string = f'{dyear}-{dmonth}-{dday}_{dhour}:{dminute}:{dsecond}'
	sw_path = os.path.join(spec.local_dir,date_string)
	print('\n\n')
	print(f'path : {sw_path}')
	writer = SummaryWriter(sw_path)
	print('\n\n')

	# tb = program.TensorBoard()
	# tb.configure(argv=[None, '--logdir', sw_path])
	# url = tb.launch()
	current_force_boundary = 100.0
	for i in range(1000000):
		result = trainer.train()
		# from IPython import embed;embed();exit()
		
		avg_force_reward = ray.get([w.foreach_env.remote(lambda env:env.get_average_force_reward()) for w in trainer.workers.remote_workers()])
		avg_force_reward = np.mean(avg_force_reward)

		ratio = 1.0
		if avg_force_reward>=0.9:
			ratio = 1.9 - avg_force_reward
		else:
			ratio = 1.0 - 0.2*(avg_force_reward-0.9)
		current_force_boundary *= ratio
		for w in trainer.workers.remote_workers():
			w.foreach_env.remote(lambda env:env.set_current_force_boundary(current_force_boundary))

		training_iteration = result['training_iteration']
		episode_len_mean = result['episode_len_mean']
		episode_reward_min = result['episode_reward_min']
		episode_reward_max = result['episode_reward_max']
		episode_reward_mean = result['episode_reward_mean']
		episodes_this_iter = result['episodes_this_iter']
		episodes_total = result['episodes_total']
		time_this_iter_s = result['time_this_iter_s']
		time_total_s = result['time_total_s']
		timesteps_total = result['timesteps_total']


		h = int(time_total_s//3600.0)
		m = int(time_total_s//60.0)
		m = m - h*60
		s = int(time_total_s)
		s = s - h*3600 - m*60
		print(f'--# {training_iteration} : {time_this_iter_s:.2f}s({h}h:{m}m:{s}s)-----------------------------------')
		print(f'episode len     --- {episode_len_mean:.2f}')
		print(f'reward          --- mean : {episode_reward_mean:.2f}, min : {episode_reward_min:.2f}, max : {episode_reward_max:.2f}')
		print(f'num episode     --- {episodes_this_iter}, total : {episodes_total}')
		print(f'num transitions --- {timesteps_total}')
		print(f'force boundary  --- {current_force_boundary:.2f}')

		writer.add_scalar('data/time',time_total_s,timesteps_total)
		writer.add_scalar('data/episode_len',episode_len_mean,timesteps_total)
		writer.add_scalar('data/reward_mean',episode_reward_mean,timesteps_total)
		writer.add_scalar('data/force_boundary',current_force_boundary,timesteps_total)

		ball_dist = ray.get([w.foreach_env.remote(lambda env:env.get_ball_distribution()) for w in trainer.workers.remote_workers()])
		ball_dist = np.array(ball_dist).squeeze()
		ball_dist = ball_dist.mean(axis=0)
		for w in trainer.workers.remote_workers():
			w.foreach_env.remote(lambda env:env.set_ball_distribution(ball_dist))

		if training_iteration % spec.checkpoint_freq == 0:
			checkpoint = trainer.save(spec.local_dir)
			print(f"checkpoint saved at {checkpoint}")
			writer.add_text('data/checkpoint_path',checkpoint,timesteps_total)
			ball_dist -= 0.0
			ball_dist /= 100.0
			
			img = np.zeros(shape=(3,ball_dist.shape[0],ball_dist.shape[1]))

			img[0] = ball_dist
			img[1] = ball_dist
			img[2] = 1.0 - ball_dist

			writer.add_image('data/force_dist',img, timesteps_total)


