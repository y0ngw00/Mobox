config = {
	'num_envs' : 64,
	# 'num_envs' : 4,
	# 'save_path' : '/home/seunghwan/Documents/ComCon/data/learning/',
	# 'save_path' : '/home/seunghwan/Documents/comcon_remote_new/data/learning/',
	'save_path' : '/home/seunghwan/Documents/comcon_remote/data/learning/',
	'save_at_start' : True,
	'model0' : {
		'sample_std' : 0.3,
		'policy_hiddens' : [256, 256],
		'policy_activations' : ['relu', 'relu', None],
		'policy_init_weights' : [0.1, 0.1, 0.01],
		'value_hiddens' : [256, 256],
		'value_activations' : ['relu', 'relu', None],
		'value_init_weights' : [0.1, 0.1, 0.01],
	},

	'model1' : {
		'sample_std' : 0.2,
		'policy_hiddens' : [16],
		'policy_activations' : ['relu', None],
		'policy_init_weights' : [0.1, 0.01],
		'value_hiddens' : [16],
		'value_activations' : ['relu', None],
		'value_init_weights' : [0.1, 0.01],
	},

	
	'policy0' : {
		'gamma' : 0.95,
		'lb' : 0.95,
		'lr' : 1e-5,
		'policy_clip' : 0.2,
		'value_clip' : 1.0,
		'grad_clip' : 0.5,
		# 'grad_clip' : 1000.0,
		'kl' : 0.01,
		'entropy' : -0.01
	},

	'policy1' : {
		'gamma' : 0.95,
		'lb' : 0.95,
		'lr' : 1e-5,
		'policy_clip' : 0.2,
		'value_clip' : 1.0,
		'grad_clip' : 0.5,
		# 'grad_clip' : 1000.0,
		'kl' : 0.01,
		'entropy' : -0.01
	},
	'trainer' : {
		'sample_size' : 2048,
		'num_sgd_iter' : 5,
		'sgd_minibatch_size' : [128,128],
		'save_iteration' : [10,500]
	}
}