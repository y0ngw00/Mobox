import collections
import numpy as np
from os import listdir
from os.path import isfile, join
from IPython import embed
import math
'''general settings'''
num_cpus = 16
num_gpus = 1
run = 'PPO'
name = 'expert'

local_dir = '/home/seunghwan/Documents/ComCon/data/learning/'
checkpoint_freq = 20
checkpoint_at_end = True
stop = {}
stop['time_total_s'] = 720000
'''config'''
project_dir = '/home/seunghwan/Documents/ComCon/python'

config = {
	'env' : 'RayEnv',
	# 'log_level' : 'WARN',
	'log_level' : 'ERROR',
	'gamma' : 0.95,
	'lambda' : 0.95,
	'clip_param' : 0.2,
	'kl_coeff' : 0.0,
	'vf_clip_param' : 1000,
	'num_sgd_iter' : 10,
	'lr' : 0.00001,
	'sgd_minibatch_size' : 128,
	'horizon' : 900,
	'train_batch_size' : 2048,
	# 'train_batch_size' : 16384,
	'framework' : 'torch',
	'num_envs_per_worker' : 1,
	'num_cpus_per_worker' : 1,
	'num_gpus_per_worker' : 0,
	'num_workers' : num_cpus,
	'num_gpus' : num_gpus,
	'rollout_fragment_length' : 32,
	'model' : {
		'custom_model': 'fc',
		'custom_model_config' :{
			'sample_std': 0.1}
	},
	'batch_mode' : 'truncate_episodes',
	'observation_filter' : 'MeanStdFilter',
	'grad_clip' : 0.5,
	'vf_loss_coeff' : 1.0,
	'normalize_actions' : False,

	'env_config' : {
		'kinematics' : False,
		'project_dir' : project_dir,
		'motion_generator_config' : {
			'use_msd' : True,
			'bvh_file' : join(project_dir,'data/motion/open_door.bvh'), 

			#order = simHips,simSpine,
			#		 simLeftArm,simLeftForeArm,simLeftHand,
			#        simHead,
			#        simRightArm,simRightForeArm,simRightHand,
			#	     simRightUpLeg,simRightLeg,simRightFoot,
			#        simLeftUpLeg,simLeftLeg,simLeftFoot
			'mass_spring_damper': {
				'modulus_of_elasticity_func' : lambda x : x,
				'stiffness_coeff' : [0.0, 15.0,
									1.0, 1.0, 0.0,
									0.0,
									1.0, 1.0, 0.0,
									0.0, 0.0, 0.0,
									0.0, 0.0, 0.0],
				'damping_coeff' : [0.0, 5.0,
								   8.0, 7.0, 5.0,
								   5.0,
								   8.0, 7.0, 5.0,
								   0.0, 0.0, 0.0,
								   0.0, 0.0, 0.0],
				'mass_coeff' : [1e8, 5.0,
								0.8, 1.0, 1e8,
								1.2,
								0.8, 1.0, 1e8,
								1e8, 1e8, 1e8,
								1e8, 1e8, 1e8],
			},
		},
		'sim_time_step' : 1.0/240.0,
		'con_time_step' : 1.0/30.0,
		'max_episode_len' : 30*7,
		'cube_urdf' : join(project_dir,'data/door.urdf'),
		'agent' : {
			'observation_scale' : 100000.0,
			'char_info_module' : join(project_dir,'data/skel_char_info.py'),
			'sim_char_file' : join(project_dir,'data/skel.urdf'),
			'self_collision' : True,
			'action_range_min' : -np.pi,
			'action_range_max' : np.pi,
			'action_range_min_policy' : -1.0,
			'action_range_max_policy' : 1.0,

			'action_range_min_cc' : -np.pi*0.1,
			'action_range_max_cc' : np.pi*0.1,
			'action_range_min_policy_cc' : -1.0,
			'action_range_max_policy_cc' : 1.0,

			'imitation_frame_window' : [5],
			# 'scale_pos' : 4.0,
			'scale_pos' : 4.0,
			'scale_vel' : 0.2,
			'scale_ee' : 1.0,
			'scale_root' : 1.0,
			'scale_com' : 1.0,
			'rho' : 0.1,
		}
	}
}


render_config = {
	'width' : 1920,
	'height' : 1080,
	'fps' : 30,
	'file_tex_ground' : 'data/image/grid2.png',
	'flag' : {
		'follow_cam' : True,
		'ground' : True,
		'origin' : False,
		'shadow' : True,
		'sim_model' : True,
		'kin_model' : True,
		'joint' : False,
		'com_vel' : False,
		'collision' : False,
		'overlay' : True,
		'target_pose' : False,
		'auto_play' : False,
		'facing_frame' : False,
		'force' : True,
	},
	'toggle' : {
		b'0' : 'follow_cam',
		b'1' : 'ground',
		b'2' : 'origin',
		b'3' : 'shadow',
		b'4' : 'sim_model',
		b'5' : 'kin_model',
		b'6' : 'joint',
		b'7' : 'com_vel',
		b'8' : 'collision',
		b'9' : 'overlay',
		b't' : 'target_pose',
		b' ' : 'auto_play',
		b'-' : 'force',
		b'F' : 'facing_frame',
		b'o' : 'obstacle',
	}
}
