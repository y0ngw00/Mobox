config = {
	'num_envs' : 16,
	'save_path' : '../data/learning/',
	'save_at_start' : True,
	'model' : {
		'sample_std' : 0.1,
		'fixed_std' : True,
		'policy_hiddens' : [1024, 512],
		'policy_activations' : ['relu', 'relu', None],
		'policy_init_weights' : [0.1, 0.1, 0.01],
		'value_hiddens' : [1024, 512],
		'value_activations' : ['relu', 'relu', None],
		'value_init_weights' : [0.1, 0.1, 0.01],
	},
	
	'policy' : {
		'gamma' : 0.95,
		'lb' : 0.95,
		'lr' : 2e-6,
		'policy_clip' : 0.2,
		'value_clip' : 1.0,
		'grad_clip' : 0.5,
		'kl' : 0.01,
		'entropy' : 0.0
	},

	'discriminator_model' : {
		'hiddens' : [1024,512,512],
		'activations' : ['relu', 'relu', 'relu', None],
		'init_weights' : [0.1, 0.1,0.1, 1.0],
	},
	
	'discriminator' : {
		'loss' : 'lsq loss',
		'w_grad' : 10.0,
		'grad_clip' : 0.5,
		'w_reg' : 0.05,
		'w_decay' : 0.0005,
		'r_scale' : 2.0,
		'w_style': 0.7,
		'w_class': 0.3,
		'lr' : 1e-8,
	},

	'trainer' : {
		'sample_size' : 4096,
		'num_sgd_iter' : 5,
		'sgd_minibatch_size' : 64,
		'num_disc_expert' : 4,
		'num_disc_sgd_iter' : 2,
		'disc_sgd_minibatch_size' : 64,
		'disc_buffer_len' : 100000,
		'save_iteration' : [10,500],
		'epsilon' : 0.05
	}

}