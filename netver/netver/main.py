import os; 
from netver.backend.ProVe import ProVe
from netver.backend.CompleteProve import CompleteProVe
from netver.backend.CountingProVe import CountingProVe
from netver.backend.MILP import MILP
from netver.backend.LinearSolver import LinearSolver
from netver.backend.Estimated import Estimated
from netver.utils.dual_net_gen import *
import tensorflow as tf; import numpy as np

# TODO: implements the sanity check method
# TODO: add support for other neural network format

class NetVer:

	"""
	Main class of the NetVer project, this project implements different methods for the formal verificaiton of neural network. Currently the support is 
	only for feed forward neural netowrk and a proeprty can be expressed in three different format, see https://github.com/d-corsi/NetworkVerifier for 
	a complete list of the supported format and algorithms.
	
	This class is the hub of the project, translates the peroperties expressed in different format in the correct format for the tools (same for the network
	models). This class also provides some check for the structure and the given parameters and returns errors and warning message if some parameters is not
	correctly setted.

	All the network/property are translated to solve the following two types of query:
		- positive: all the outputs of the network must be greater than 0
		- reverse positive: at least one output of the network must be greater than 0

	Attributes
	----------
		verifier : Object
			the requested verification tool correctly setted for the verification
		algorithms_dictionary: dict
			a dictionary that translate the key_string of the methods inside the object for the tool
		
	Methods
	-------
		run_verifier( verbose )
			method that formally verify the property P on the ginve network, running the given verification tool

	"""

	# Dictionary for the translation of the key string to the requested alogirhtm class object
	algorithms_dictionary = {
		"prove" : ProVe,
		"complete_prove" : CompleteProVe,
		"counting_prove": CountingProVe,
		"MILP" : MILP,
		"linear" : LinearSolver,
		"estimated" : Estimated
	}


	def __init__( self, algo, network, property, **kwargs ):

		"""
		Constructor of the class. This method builds the object verifier, setting all the parameters and parsing the proeprty 
		and the network to the correct format for the tool. 

		Parameters
		----------
			algo : string
				a string that indicates the algorith/tool to use, a list of all the available keys here https://github.com/d-corsi/NetworkVerifier
			property : dict
				a dictionary that describe the property to analyze, overview of the structure here https://github.com/d-corsi/NetworkVerifier
			network : tf.keras.Model
				neural network model to analyze
			kwargs : **kwargs
				dictionary to overide all the non-necessary paramters (if not specified the algorithm will use the default values)	
		"""

		if property["type"] == "positive":
			self.primal_network = network
			kwargs["dual_network"] = create_dual_net_positive( network )
			kwargs["reversed"] = False

		elif property["type"] == "PQ":
			self.primal_network = self._create_net_PQ( network, property )
			kwargs["dual_network"] = create_dual_net_PQ( network, property ) 
			kwargs["reversed"] = False

		elif property["type"] == "decision":
			self.primal_network = self._create_net_decision( network, property )
			kwargs["dual_network"] = create_dual_net_positive( self.primal_network )
			kwargs["reversed"] = True

		else:
			raise ValueError("Invalid property type, valid values: [positive, PQ, decision]")

		# Check mismatch between size of the input layer and domain P of the property
		assert( self.primal_network.input.shape[1] == len(property["P"]) )

		# Creation of the object verifier, calling the selected algorithm class with the required parameters
		self.verifier = self.algorithms_dictionary[algo]( self.primal_network, np.array(property["P"]), **kwargs )

		
	def run_verifier( self, verbose=0 ):

		"""
		Method that perform the formal analysis, launching the object verifier setted in the constructor.

		Parameters
		----------
			verbose : int
				when verbose > 0 the software print some log informations

		Returns:
		--------
			sat : bool
				true if the proeprty P is verified on the given network, false otherwise
			info : dict
				a dictionary that contains different information on the process, the 
				key 'counter_example' returns the input configuration that cause a violation
				key 'exit_code' returns the termination reason (timeout or completed)
		"""
		
		#
		return self.verifier.verify( verbose )


	def _create_net_decision( self, network, property ):

		"""
		This method modify the network using the given network and the decision property (i.e., the pivot node can not be the one with the highest value), 
		to create a network ehich is verifiable with a 'reverse positive' query (i.e., at least one output of the network must be greater than 0). 
		To this end, the method adds n-1 nodes to the netwrok, each of which is the results of itself - the pivot node.
		If one of the other output is greater than the pivot node the 'reverse positive' query is succesfully proved.

		Parameters
		----------
			network : tf.keras.Model
				tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
			property : dict
				a dictionary that describe the 'decision' property to analyze (overview of the structure here https://github.com/d-corsi/NetworkVerifier)	

		Returns:
		--------
			network_custom : tf.keras.Model
				the netowrk model modified for the 'reverse positive' query
		"""

		# Get the size of the output layer and the pivot node
		output_size = network.output.shape[1]
		prp_node = property["A"]

		# Create the custom last layer (linear activation) of n-1 nodes, 
		# and create the fully connected new network with this last layer attached
		output_custom = tf.keras.layers.Dense(output_size-1, activation='linear', name='output_custom')(network.output)
		network_custom = tf.keras.Model( network.input, output_custom )

		# Create the array for the biases and weights of the new fully connected layer to zero
		custom_biases = np.zeros(output_size-1)
		custom_weights = np.zeros((output_size, output_size-1))

		# Set to -1 the weights exiting from the pivot node for the formula node_i - pivot
		for i in range(output_size-1): custom_weights[prp_node][i] = -1

		# To complete the formula node_i - pivot set to 1 the exit weights of the i-th node
		c = 0
		for i in range(output_size):
			if i == prp_node: continue
			custom_weights[i][c] = 1
			c += 1

		# Set the weights and biases of the last fully connectd layer to the new generated values
		network_custom.layers[-1].set_weights([custom_weights, custom_biases])

		#
		return network_custom


	def _create_net_PQ( self, network, property ):

		"""
		This method modify the netowrk netowrk using the given network and the PQ property (i.e., each output must be inside the corresponding bound given by the property Q), 
		to create a netowrk that is verifiable with a 'positive' query (i.e., at least one output of the network must be greater than 0). 
		To this end, the method adds 2n nodes to the netwrok, each of which is the results of node +/- is upper and lower bound.
		If each node is inside its bound the 'positive' query is succesfully proved.

		Parameters
		----------
			network : tf.keras.Model
				tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
			property : dict
				a dictionary that describe the 'PQ' property to analyze (overview of the structure here https://github.com/d-corsi/NetworkVerifier)	

		Returns:
		--------
			network_custom : tf.keras.Model
				the netowrk model modified for the 'reverse positive' query
		"""

		# Get the size of the output layer and the pivot node
		output_size = len(property["Q"])

		# Create the custom last layer (linear activation) of 2n nodes, 
		# and create the fully connected new network with this last layer attached
		output_custom = tf.keras.layers.Dense(output_size*2, activation='linear', name='output_custom')(network.output)
		network_custom = tf.keras.Model( network.input, output_custom )

		# Create the array for the biases and weights of the new fully connected layer
		# the biases have the value of the bounds for each node to generate the formula
		# n-lower_bound and n+lower_bound
		custom_biases = np.array([[-a, b] for a, b in property["Q"]]).flatten()
		custom_weights = np.zeros((output_size, output_size*2))
		
		# Set the weight exiting from the output node to the new generated layer to 1 or -1 
		# to concretize the fotmula for the positive query
		for i in range(output_size):
			custom_weights[i][i*2+0] = 1
			custom_weights[i][i*2+1] = -1

		# Set the weights and biases of the last fully connectd layer to the new generated values
		network_custom.layers[-1].set_weights([custom_weights, custom_biases])

		#
		return network_custom


	def _sanity_check( self, algo, network, property ):

		"""
		Constructor of the class. This method builds the object verifier, setting all the parameters and parsing the proeprty 
		and the network to the correct format for the tool. 

		Parameters
		----------
			algo : string
				a string that indicates the algorith/tool to use, a list of all the available keys here https://github.com/d-corsi/NetworkVerifier
			property : dict
				a dictionary that describe the property to analyze, overview of the structure here https://github.com/d-corsi/NetworkVerifier
			network : tf.keras.Model
				neural network model to analyze

		Returns:
		--------
			sanity_check : bool
				return True if the sanity check did not find any errors, False otherwise
		"""

		# TODO: check the network structure and the activations [relu, sigmoid, tanh, linear]
		# TODO: check thah the property Q and P comes with the same size of the network (assert model.output.shape[1] == len(property["Q"]))
		# TODO: add a warning if launch MILP/Linear on a big network
		sanity_check = True

		#
		return sanity_check


