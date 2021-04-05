import argparse
import importlib.util
import datetime
import os
import threading

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pycomcon
import model
import ppo
import trainer

from torch.utils.tensorboard import SummaryWriter
cuda = torch.cuda.is_available()

def time_to_hms(t):
	h = int((t)//3600.0)
	m = int((t)//60.0)
	s = int((t))
	m = m - h*60
	s = (t)
	s = s - h*3600 - m*60
	return h,m,s
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', required=False, type=str)
	parser.add_argument('--config', required=True, type=str)
	parser.add_argument("--checkpoint", type=str, default=None)
	args = parser.parse_args()
	

	spec = importlib.util.spec_from_file_location("config", args.config)
	spec_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(spec_module)
	spec = spec_module

	config = spec.config

	dyear = datetime.datetime.now().year
	dmonth = datetime.datetime.now().month
	dday = datetime.datetime.now().day
	dhour = datetime.datetime.now().hour
	dminute = datetime.datetime.now().minute
	dsecond = datetime.datetime.now().second
	date_string = f'{dyear}-{dmonth}-{dday}_{dhour}:{dminute}:{dsecond}'
	if args.name is not None:
		sw_path = os.path.join(config['save_path'],args.name)
	else:
		sw_path = os.path.join(config['save_path'],date_string)
	print('\n')
	print(f'tensorboard --logdir={sw_path}')
	writer = SummaryWriter(sw_path)
	print('\n')
	def launchTensorBoard():
		os.system('tensorboard --logdir=' + sw_path)
		return
	# t = threading.Thread(target=launchTensorBoard, args=([]))
	# t.start()
	# config = config
	
	''' Make Environment '''
	envs = pycomcon.vector_env(config['num_envs'])
	
	''' Define Model '''
	md = model.FCModel(envs.get_dim_state(),envs.get_dim_action(),config['model'])
	forcemd = model.ElementWiseFCModel(envs.get_dim_state(), envs.get_dim_statef(), envs.get_dim_actionf(), config['modelf'])

	''' Define policy '''	
	policy = ppo.FCPolicy(md, config['policy'])
	policyf = ppo.ElementWiseFCPolicy(forcemd, config['policyf'])
	
	''' Define Trainer '''
	trainer = trainer.Trainer(envs, policy, policyf, config['trainer'])
	if args.checkpoint is not None:
		trainer.load(args.checkpoint) 
	cnt = 0
	trainer.save(sw_path)
	while True:
		trainer.generate_samples()
		trainer.compute_TD_GAE()
		
		trainer.concat_samples()
		trainer.concat_samplefs()
		trainer.standarize_samples('ADVANTAGES')
		envs.sync_envs()
		trainer.optimize()
		log = trainer.optimizef()

		h,m,s=time_to_hms(log['elapsed_time'])
		print('# {}, {}h:{}m:{:.1f}s - '.format(log['num_iterations_so_far'],h,m,s))
		print('imitation len : {:.1f}, rew : {:.1f}, std : {:.3f} samples : {:,}'.format(log['mean_episode_len'], log['mean_episode_reward'], log['std'], log['num_samples_so_far']))
		print('force     len : {:.1f}, rew : {:.1f}, std : {:.3f} samples : {:,}'.format(log['mean_episodef_len'], log['mean_episodef_reward'], log['stdf'], log['num_samplefs_so_far']))

		''' logging and save '''
		writer.add_scalar('data/time',log['elapsed_time'],log['num_samples_so_far'])
		writer.add_scalar('data/episode_len',log['mean_episode_len'],log['num_samples_so_far'])
		writer.add_scalar('data/reward_mean',log['mean_episode_reward'],log['num_samples_so_far'])
		writer.add_scalar('data/std',log['std'],log['num_samples_so_far'])

		writer.add_scalar('data/episodef_len',log['mean_episodef_len'],log['num_samplefs_so_far'])
		writer.add_scalar('data/rewardf_mean',log['mean_episodef_reward'],log['num_samplefs_so_far'])
		writer.add_scalar('data/stdf',log['stdf'],log['num_samplefs_so_far'])

		trainer.save(sw_path)

# states = envs.get_states()
# states = torch.tensor(states)

# states = states.type(torch.FloatTensor)
# if cuda:
# 	states = states.cuda()

# actions = model(states)