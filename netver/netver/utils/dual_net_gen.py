import tensorflow as tf; import numpy as np	
	

"""
A collection of method for the generation of the dual networks for Complete Prove.

dual_network: tf.keras.Model
	the dual netowrk is built to deny the properties, is the negation of the main netowrk, a property is violated when "at least
	ONE output of the dual netowrk is greater than zero. It also works in "reverse" mode, a property is violated when 
	"ALL the outputs of the dual network is greater than zero"
"""


def create_dual_net_PQ( network, property ):

	"""
	This method generate the dual netowrk using the given network and the PQ property (i.e., each output must be inside the corresponding bound given by the property Q), 
	
	Parameters
	----------
		network : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
		property : dict
			a dictionary that describe the 'PQ' property to analyze (overview of the structure here https://github.com/d-corsi/NetworkVerifier)	

	Returns:
	--------
		dual_network : tf.keras.Model
			the netowrk model modified for the 'reverse positive' query
	"""

	# Get the size of the output layer
	output_size = len(property["Q"])

	# Create the custom last layer (linear activation) of 2n nodes, 
	# and create the fully connected new network with this last layer attached
	output_custom = tf.keras.layers.Dense(output_size*2, activation='linear')(network.output)
	dual_network = tf.keras.Model( network.input, output_custom )

	# Create the array for the biases and weights of the new fully connected layer
	# the biases have the value of the bounds for each node to generate the formula
	# n-lower_bound and n+lower_bound
	# NB: For the dual the formula is the reverse of the original
	custom_biases = np.array([[a, -b] for a, b in property["Q"]]).flatten()
	custom_weights = np.zeros((output_size, output_size*2))
	
	# Set the weight exiting from the output node to the new generated layer to 1 or -1 
	# to concretize the fotmula for the positive query
	# NB: For the dual the formula is the reverse of the original
	for i in range(output_size):
		custom_weights[i][i*2+0] = -1
		custom_weights[i][i*2+1] = 1

	# Set the weights and biases of the last fully connectd layer to the new generated values
	dual_network.layers[-1].set_weights([custom_weights, custom_biases])

	#	
	return dual_network


def create_dual_net_positive( network ):

	"""
	This method generate the dual netowrk using the given network and the decision property (i.e., the pivot node can not be the one with the highest value),
	
	Parameters
	----------
		network : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format

	Returns:
	--------
		dual_network : tf.keras.Model
			the netowrk model modified for the 'reverse positive' query	
	"""

	# Get the size of the output layer
	output_size = network.output.shape[1]

	# Create the custom last layer (linear activation) of n nodes, 
	# and create the fully connected new network with this last layer attached
	output_custom = tf.keras.layers.Dense(output_size, activation='linear')(network.output)
	dual_network = tf.keras.Model( network.input, output_custom )

	# Create the array for the biases and weights of the new fully connected layer to zero
	custom_biases = np.zeros(output_size)
	custom_weights = np.zeros((output_size, output_size))

	# Set to -1 the weights exiting from each output node to create the negation of the output layer
	for i in range(output_size): custom_weights[i][i] = -1
	
	# To complete the formula node_i - pivot set to 1 the exit weights of the i-th node
	dual_network.layers[-1].set_weights([custom_weights, custom_biases])

	#
	return dual_network

