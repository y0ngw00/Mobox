config = {
	'num_envs' : 64,
	# 'num_envs' : 4,
	# 'save_path' : '/home/seunghwan/Documents/ComCon/data/learning/',
	# 'save_path' : '/home/seunghwan/Documents/comcon_remote_new/data/learning/',
	'save_path' : '/home/seunghwan/Documents/comcon_remote/data/learning/',
	'model' : {
		'sample_std' : 0.3,
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
		'policy_clip' : 0.2,
		'value_clip' : 1.0,
		'grad_clip' : 0.5,
		# 'grad_clip' : 1000.0,
		'kl' : 0.01,
		'entropy' : 0.01
	},
	'modelf' : {
		'sample_std' : 0.3,
		'policy0_hiddens' : [256, 64],
		'policy0_activations' : ['relu', None],
		'policy0_init_weights' : [0.1, 0.1],
		'policy1_hiddens' : [64, 64],
		'policy1_activations' : ['relu', 'relu', None],
		'policy1_init_weights' : [0.1, 0.1, 0.01],

		'value0_hiddens' : [256, 64],
		'value0_activations' : ['relu', 'relu', None],
		'value0_init_weights' : [0.1, 0.1, 0.01],
		'value1_hiddens' : [64, 64],
		'value1_activations' : ['relu', 'relu', None],
		'value1_init_weights' : [0.1, 0.1, 0.01],
	},
	'policyf' : {
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
		'sgd_minibatch_size' : 128,
		'sgd_minibatch_sizef' : 128,
		'save_iteration' : 10,
		'save_iteration2' : 500,
	}
}