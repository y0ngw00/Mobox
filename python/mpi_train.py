import argparse
import importlib.util
import datetime
import os
import threading
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pycomcon
import ppo
import discriminator
import mpi_trainer

from mpi4py import MPI

from torch.utils.tensorboard import SummaryWriter


def load_config(path):
	spec = importlib.util.spec_from_file_location("config", path)
	spec_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(spec_module)
	spec = spec_module

	return spec.config

def save_config(path, config):
	folder_path = path
	if not os.path.exists(path):
			os.mkdir(path)

	with open(path+'/config.py', 'w') as f:
		f.write('config = {\n')
		for name, value in config.items():
			if type(value) == dict :
				f.write('\t\'{}\' : {{\n'.format(name))
				for names, values in value.items():
					f.write('\t\t\'{}\' : {!r}\n'.format(names, values))
				f.write('\t}\n')
			else :
				f.write('\t\'{}\' : {!r}\n'.format(name, value))

		f.write('}')


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
	
	checkpoint = args.checkpoint
	config = load_config(args.config)
	save_path = define_save_path(args.name)
	save_config(save_path, config)

	if checkpoint is not None:
		meta_info = checkpoint
		meta_info.replace('current.pt','')
		config = load_config(args.checkpoint + 'config.py')

	trainer = mpi_trainer.Trainer(pycomcon.env, config, save_path)
	
	if checkpoint is not None:
		trainer.load(checkpoint)
	done = False
	try:
		while not done:
			trainer.step()
	except KeyboardInterrupt:
		print('abort mpi')
		MPI.COMM_WORLD.Abort(1)
