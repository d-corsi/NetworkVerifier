from xml import dom
import numpy as np


def multi_area_estimator( input_domain, net_model, estimation_precision=100 ):

	domains = [np.random.uniform(input_area[:, 0], input_area[:, 1], (estimation_precision, 2) ) for input_area in input_domain]
	network_input = np.array(domains).reshape( estimation_precision*input_domain.shape[0], -1 )

	network_output = net_model(network_input).numpy().reshape( input_domain.shape[0], estimation_precision, -1 )

	lower_bounds = np.min( network_output, axis=1 ).reshape(-1, 1)
	upper_bounds = np.max( network_output, axis=1 ).reshape(-1, 1)

	reshaped_bound = np.hstack([lower_bounds, upper_bounds]).reshape( input_domain.shape[0], -1, 2)
	return reshaped_bound
	

def multi_area_propagation_cpu( input_domain, net_model ):
	test_codomain = np.array([ single_area_propagation_cpu(d, net_model) for d in input_domain ])
	return test_codomain


def single_area_propagation_cpu( input_domain, net_model ):

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