config = {
	'num_envs' : 16,
	# 'num_envs' : 4,
	# 'save_path' : '/home/seunghwan/Documents/ComCon/data/learning/',
	# 'save_path' : '/home/seunghwan/Documents/comcon_remote_new/data/learning/',
	'save_path' : '/home/seunghwan/Documents/comcon_remote/data/learning/',
	'save_at_start' : True,
	'model' : {
		'sample_std' : 0.1,
		'fixed_std' : True,
		'policy_hiddens' : [256, 256],
		'policy_activations' : ['relu', 'relu', None],
		'policy_init_weights' : [0.1, 0.1, 0.01],
		'value_hiddens' : [256, 256],
		'value_activations' : ['relu', 'relu', None],
		'value_init_weights' : [0.1, 0.1, 0.01],
	},
	
	'policy' : {
		'gamma' : 0.95,
		'lb' : 0.95,
		'lr' : 1e-5,
		# 'lr' : 1e-4,
		'policy_clip' : 0.2,
		'value_clip' : 1.0,
		'grad_clip' : 0.5,
		'kl' : 0.01,
		'entropy' : 0.0
	},

	'discriminator_model' : {
		'hiddens' : [256, 256],
		'activations' : ['relu', 'relu', None],
		'init_weights' : [0.1, 0.1, 1.0],
	},
	
	'discriminator' : {
		# 'w_grad' : 10.0,
		'w_grad' : 10.0,
		'grad_clip' : 0.5,
		'w_reg' : 0.05,
		'w_decay' : 0.0005,
		'r_scale' : 2.0,
		'lr' : 5e-6,
		# 'lr' : 1e-4,
	},

	'trainer' : {
		'sample_size' : 4096,
		# 'sample_size' : 2048,
		'num_sgd_iter' : 3,
		'sgd_minibatch_size' : 256,
		# 'sgd_minibatch_size' : 128,
		# 'sgd_minibatch_size' : 32,
		
		'num_disc_sgd_iter' : 2,
		# 'disc_sgd_minibatch_size' : 16,
		'disc_sgd_minibatch_size' : 16,
		'disc_buffer_len' : 100000,

		'save_iteration' : [10,500],
	}
}