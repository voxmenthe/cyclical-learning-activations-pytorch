hyperparams = {
	'model_list': [
        # 'squeezenet1_0',
        # 'squeezenet1_1',
        # 'squeezenet1_0_act',
        # 'squeezenet1_1_act',
        'DPN92_act3',
        'PreActResNet101_act',      
        'wrn92_2_act',
        'ResNeXt29_4x64d_act',
        #'PNASNetA_act',
        'PNASNetB_act',
        'DPN92_act2',
        'DenseNet161_act',
        #'DPN92',
	],

	'activation_list': [
        'swishleak_ns_006',
        'eswish_1_8',
        'F.softsign',
        'F.relu',
        #'F.selu',
        #'F.elu',
        #'F.leaky_relu'
        #'pennington1',
        #'pennington2',
        #'b_swishleak_ns_017',
        # 'swishleak_ns_003',
        # 'swishleak_ns_032',
        # 'eswish_1_7',
        'leaky_relu_ns_006',
        # 'bidrelmomv2_tv11_m06',
        # 'bidrelmomv2_tv16_m12',
        'b_swishleak_ns_003',
        # 'b_swishleak_ns_006',
        # 'b_swishleak_ns_017',
        # 'b_swishleak_ns_032',
        # 'b_eswish_1_7',
        # 'b_eswish_1_8',
        # 'b_bidrelmomv2_tv23',
	],

	# each optimizer needs to be paired with an appropriate starting learning rate
	# if you're using default settings, but these lrs are ignored in in-epoch lr and cycle_lr_list
	'optimizer_list': [
		('AddSign',0.17), 
		('PowerSign', 0.17),
		('optim.SGD',0.32),
		#('optim.Adam',0.05), 
		#('optim.Adagrad', 0.15),
		#('optim.RMSprop', 0.08)

	],

	# different random seeds to try
	'random_seed_list': [123], #[42, 999] etc

	# number of times to repeat each permutation - for testing purposes only
	'num_repeats': 1,

	# used with in-epoch lr decay - the entries in this file overwrite the command line defaults
	'inepoch_max': 0.21,
	'inepoch_min': 0.12,
	'inepoch_max_decay': 0.99,  # optional decay factor - comment out if not wanted
	'inepoch_min_decay': 0.955,  # optional decay factor - comment out if not wanted
	'inepoch_batch_step': None,
	# used in the cycle_lr_list learning rate schedule - enter as a string-list
	# comment out here to use the default
	#'cycle_lr_list': '[0.54, 0.32, 0.24, 0.16, 0.08, 0.05, 0.01]'
	#'cycle_lr_list': '[0.26, 0.23, 0.13, 0.1, 0.19, 0.09, 0.03]'
	#'cycle_lr_list': '[0.24, 0.09, 0.04]'
	'notes': "If you set max_decay less than min_decay, inepoch_min can eventually exceed inepoch_max, which is kind of weird, but should still work - I haven't tested extreme values though."
}

"""
Possible activation choices:
        'swishleak_ns_003',
        'swishleak_ns_006',
        'swishleak_ns_032',
        'eswish_1_7',
        'eswish_1_8',
        'leaky_relu_ns_006',
        'bidrelmomv2_tv11_m06',
        'bidrelmomv2_tv16_m12',
        'b_swishleak_ns_003',
        'b_swishleak_ns_006',
        'b_swishleak_ns_017',
        'b_swishleak_ns_032',
        'b_eswish_1_7',
        'b_eswish_1_8',
        'b_bidrelmomv2_tv23',
        'F.softsign',
        'F.relu',
        'F.leaky_relu',

Possible model choices (with flexible activations):
	'DPN92_act1',
	'DPN92_act2',
     'PreActResNet101_act',
     'DenseNet161_act',
     'PNASNetA_act',
     'PNASNetB_act',
     'wrn92_2_act',
     'ResNeXt29_4x64d_act',



Possible optimizer choices:
		('AddSign',0.24), 
		('PowerSign', 0.24),
		('optim.SGD',0.24), 
		('optim.Adadelta', 0.24),
		('optim.Adagrad', 0.24),
		('optim.Adam',0.24), 
		('optim.Adamax',0.24),
		('optim.ASGD',0.24),
		('optim.LBFGS',0.24),
		('optim.RMSprop',0.24),
		('optim.Rprop',0.24),
		('optim.SparseAdam',0.24), 
"""