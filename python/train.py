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
import discriminator
import trainerAMP

from torch.utils.tensorboard import SummaryWriter
cuda = torch.cuda.is_available()


def load_config(path):
	spec = importlib.util.spec_from_file_location("config", path)
	spec_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(spec_module)
	spec = spec_module

	return spec.config

def define_save_path(name):
	dyear = datetime.datetime.now().year
	dmonth = datetime.datetime.now().month
	dday = datetime.datetime.now().day
	dhour = datetime.datetime.now().hour
	dminute = datetime.datetime.now().minute

	date_string = f'{dyear}-{dmonth}-{dday}-{dhour}-{dminute}'
	if name is not None:
		savepath = os.path.join(config['save_path'],name)
	else:
		savepath = os.path.join(config['save_path'],date_string)

	return savepath

def create_summary_writer(path, launch=False):
	print(f'tensorboard --logdir={save_path}')
	writer = SummaryWriter(path)

	if launch:
		def launchTensorBoard():
			os.system('tensorboard --logdir=' + sw_path)
			return
		t = threading.Thread(target=launchTensorBoard, args=([]))
		t.start()
	return writer

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', required=False, type=str)
	parser.add_argument('--config', required=True, type=str)
	parser.add_argument("--checkpoint", type=str, default=None)
	args = parser.parse_args()
	
	config = load_config(args.config)
	save_path = define_save_path(args.name)
	writer = create_summary_writer(save_path,launch=False)

	envs = pycomcon.vector_env(config['num_envs'])
	state_experts = envs.get_states_AMP_expert()

	policy_model = model.FCModel(envs.get_dim_state(), envs.get_dim_action(), config['model'])
	discriminator_model = model.FC(envs.get_dim_state_AMP(), config['discriminator_model'])
	
	policy = ppo.FCPolicy(policy_model, config['policy'])
	discriminator = discriminator.FCDiscriminator(discriminator_model, state_experts, config['discriminator'])

	trainer = trainerAMP.TrainerAMP(envs, policy, discriminator, config['trainer'])

	if args.checkpoint is not None:
		trainer.load(args.checkpoint)
	if config['save_at_start']:
		trainer.save(save_path)

	while True:
		trainer.generate_samples()
		trainer.compute_TD_GAE()
		trainer.concat_samples()
		trainer.standarize_samples('ADVANTAGES')
		trainer.optimize()
		trainer.print_log(writer)
		trainer.save(save_path)
