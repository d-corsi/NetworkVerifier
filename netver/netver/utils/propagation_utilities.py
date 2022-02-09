import imp
import numpy as np; import tensorflow as tf


def multi_area_estimator( input_domain, net_model, estimation_precision=100 ):

	"""
	Estimate the output bound for each given domain. This method provide an UNDERESTIMATION and the bound are computed
	with a sampling process. A point cloud is sampled from the input domain and propagate through the network,
	maximum and minimum for each output node are then calculated.

	Parameters
	----------
		input_domain : list 
			the input domain expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
			(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
		net_model : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
		estimation_precision : int
			the size of the point cloud, the number of sampled point (default: 100)

	Returns:
	--------
		reshaped_bound : list
			the propagated bound in the same format of the input domain (3-dim)
	"""

	# Sampling the point cloud from the input domain and reshaped according to the tensorflow format
	domains = [np.random.uniform(input_area[:, 0], input_area[:, 1], (estimation_precision, 2) ) for input_area in input_domain]
	network_input = np.array(domains).reshape( estimation_precision*input_domain.shape[0], -1 )

	# Propagation of the input through the network
	network_output = net_model(network_input).numpy().reshape( input_domain.shape[0], estimation_precision, -1 )

	# Exraction of the lower and upper bound for each element of the input domain
	lower_bounds = np.min( network_output, axis=1 ).reshape(-1, 1)
	upper_bounds = np.max( network_output, axis=1 ).reshape(-1, 1)

	# Reshaping of the output bound according to the required format
	reshaped_bound = np.hstack([lower_bounds, upper_bounds]).reshape( input_domain.shape[0], -1, 2)

	#
	return reshaped_bound
	

def multi_area_propagation_cpu( input_domain, net_model ):

	"""
	Propagation of the input domain through the network to obtain the OVERESTIMATION of the output bound. 
	The process is performed applying the linear combination node-wise and the necessary activation functions.
	The process is on CPU, without any form of parallelization. 
	This function iterate over the function "single_area_propagation_cpu" that compute the propagation
	of a single input domain.

	Parameters
	----------
		input_domain : list 
			the input domain expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
			(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
		net_model : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format

	Returns:
	--------
		reshaped_bound : list
			the propagated bound in the same format of the input domain (3-dim)
	"""

	# Iterate over every single domain of the input domain list and call the single_area_propagation_cpu function
	reshaped_bound = np.array([ single_area_propagation_cpu(d, net_model) for d in input_domain ])

	#
	return reshaped_bound


def single_area_propagation_cpu( input_domain, net_model ):

	"""
	Implementation of the real propagation of a single bound.
	Auxiliary function for the main 'multi_area_propagation_cpu' function.

	Parameters
	----------
		input_domain : list 
			the input domain expressed as a 2-dim matrix. (a) a list of bound for each input node and 
			(b) a list of two element for the node, lower and upper
		net_model : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format

	Returns:
	--------
		entering : list
			the propagated bound in the same format of the input domain (2-dim)
	"""

	# Extract all the parameters of the network for the propagation
	weights = [ layer.get_weights()[0].T for layer in net_model.layers[1:] ]
	biases = [ layer.get_weights()[1] for layer in net_model.layers[1:]  ]
	activations = [ layer.activation for layer in net_model.layers[1:] ]

	# The entering values of the first iteration are the input for the propagation
	entering = input_domain

	# Iteration over all the layer of the network for the propagation
	for layer_id, layer in enumerate(weights):

		# Pre-computation for the linear propagation of the bounds
		max_ = np.maximum(layer, 0)
		l = entering[:, 0]

		# Pre-computation for the linear propagation of the bounds
		min_ = np.minimum(layer, 0)
		u = entering[:, 1]
		
		# Update the linear propagation with the standard formulation [Liu et. al 2021]
		l_new = np.sum( max_ * l + min_ * u, axis=1 ) + biases[layer_id]
		u_new = np.sum( max_ * u + min_ * l, axis=1 ) + biases[layer_id]
		
		# Check and apply the activation function 
		l_new = activations[layer_id](l_new)
		u_new = activations[layer_id](u_new)

		# Reshape of the bounds for the next iteration
		entering = np.concatenate( [np.vstack(l_new), np.vstack(u_new)], axis=1 )

	#
	return entering



def multi_area_propagation_gpu(input_domain, net_model, thread_number=32):

	"""
	Propagation of the input domain through the network to obtain the OVERESTIMATION of the output bound. 
	The process is performed applying the linear combination node-wise and the necessary activation functions.
	The process is on GPU, completely parallelized on NVIDIA CUDA GPUs and c++ code. 

	Parameters
	----------
		input_domain : list 
			the input domain expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
			(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
		net_model : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
		thread_number : int
			number of CUDA thread to use for each CUDA block, the choice is free and does not effect the results, 
			can however effect the performance

	Returns:
	--------
		reshaped_bound : list
			the propagated bound in the same format of the input domain (3-dim)
	"""

	# Import the necessary library for the parallelization (Cupy) and also the c++ CUDA code.
	import cupy as cp
	from netver.utils.cuda_code import cuda_code

	# Load network shape, activations and weights
	layer_sizes = []
	activations = []
	full_weights = np.array([])
	full_biases = np.array([])

	# Iterate on each layer of the network, exluding the input (tf2 stuff)
	for layer in net_model.layers[1:]:

		# Obtain the activation function list
		if layer.activation == tf.keras.activations.linear: activations.append(0)
		elif layer.activation == tf.keras.activations.relu: activations.append(1)
		elif layer.activation == tf.keras.activations.tanh: activations.append(2)
		elif layer.activation == tf.keras.activations.sigmoid: activations.append(3)

		# Obtain the netowrk shape as a list
		layer_sizes.append(layer.input_shape[1])

		# Obtain all the weights for paramters and biases
		weight, bias = layer.get_weights()
		full_weights = np.concatenate((full_weights, weight.T.reshape(-1)))
		full_biases = np.concatenate((full_biases, bias.reshape(-1)))

	# Fixe last layer size
	layer_sizes.append( net_model.output.shape[1] )

	# Initialize the kernel loading the CUDA code
	my_kernel = cp.RawKernel(cuda_code, 'my_kernel')

	# Convert all the data in cupy array beore the kernel call
	max_layer_size = max(layer_sizes)
	results_cuda = cp.zeros(layer_sizes[-1] * 2 * len(input_domain), dtype=cp.float32)
	layer_sizes = cp.array(layer_sizes, dtype=cp.int32)
	activations = cp.array(activations, dtype=cp.int32)
	input_domain = cp.array(input_domain, dtype=cp.float32)
	full_weights = cp.array(full_weights, dtype=cp.float32)
	full_biases = cp.array(full_biases, dtype=cp.float32)
	
	# Define the number of CUDA block
	block_number = int(len(input_domain) / thread_number) + 1

	# Create and launch the kernel, wait for the sync of all threads
	kernel_input = (input_domain, len(input_domain), layer_sizes, len(layer_sizes), full_weights, full_biases, results_cuda, max_layer_size, activations)
	my_kernel((block_number, ), (thread_number, ), kernel_input)
	cp.cuda.Stream.null.synchronize()

	# Reshape the results and convert in numpy array
	reshaped_bound = cp.asnumpy(results_cuda).reshape((len(input_domain), net_model.layers[-1].output_shape[1], 2))

	#
	return reshaped_bound

	