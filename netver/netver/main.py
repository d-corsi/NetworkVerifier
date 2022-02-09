import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from netver.backend.ProVe import ProVe
from netver.backend.CompelteProve import CompleteProVe
from netver.backend.MILP import MILP
from netver.backend.LinearSolver import LinearSolver
import tensorflow as tf; import numpy as np

# TODO: check the network structure and the activations [relu, sigmoid, tanh, linear]
# TODO: check thah the property Q and P comes with the same size of the network (assert model.output.shape[1] == len(property["Q"]))
# TODO: add a warning if launch MILP/Linear on a big network
# TODO: hide CuPy warning
# TODO: add more neural networks format with the translations

class NetVer:

	"""
	Main class of the NetVer project, this project implements different methods for the formal verificaiton of neural network. Currently the support is 
	only for feed forward neural netowrk and a proeprty can be expressed in three different format, see https://github.com/d-corsi/NetworkVerifier for 
	a complete list of the supported format and algorithms.
	
	This class is the hub of the project, translates the peroperties expressed in different format in the correct format for the tools (same for the network
	models). This class also provides some check for the structure and the given parameters and returns errors and warning message if some parameters is not
	correctly setted.

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
		"MILP" : MILP,
		"linear" : LinearSolver
	}


	def __init__( self, algo, network, property, **kwargs ):

		"""
        Constructor of the class.

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
			self.dual_network = self._create_dual_net_positive( network )
			self.P = np.array(property["P"])
			self.reversed = False

		elif property["type"] == "PQ":
			self.primal_network = self._create_net_PQ( network, property )
			self.dual_network = self._create_dual_net_PQ( network, property )
			self.reversed = False

		elif property["type"] == "decision":
			self.primal_network = self._create_net_decision( network, property )
			self.dual_network = self._create_dual_net_positive( self.primal_network )
			self.reversed = True

		else:
			raise ValueError("Invalid property type, valid values: [positive, PQ, decision]")

		self.verifier = self.algorithms_dictionary[algo]( self.primal_network, np.array(property["P"]), reversed=self.reversed, dual_network=self.dual_network, **kwargs )

		
	def run_verifier( self, verbose=0 ):
		
		#
		return self.verifier.verify( verbose )


	def _create_net_decision( self, network, property ):
		output_size = network.output.shape[1]
		prp_node = property["A"]

		output_custom = tf.keras.layers.Dense(output_size-1, activation='linear')(network.output)
		network_custom = tf.keras.Model( network.input, output_custom )

		custom_biases = np.zeros(output_size-1)
		custom_weights = np.zeros((output_size, output_size-1))

		for i in range(output_size-1):
			custom_weights[prp_node][i] = -1

		c = 0
		for i in range(output_size):
			if i == prp_node: continue
			custom_weights[i][c] = 1
			c += 1

		network_custom.layers[-1].set_weights([custom_weights, custom_biases])

		return network_custom


	def _create_net_PQ( self, network, property ):
		output_size = len(property["Q"])

		output_custom = tf.keras.layers.Dense(output_size*2, activation='linear')(network.output)
		network_custom = tf.keras.Model( network.input, output_custom )

		custom_biases = np.array([[-a, b] for a, b in property["Q"]]).flatten()
		custom_weights = np.zeros((output_size, output_size*2))
		
		for i in range(output_size):
			custom_weights[i][i*2+0] = 1
			custom_weights[i][i*2+1] = -1

		network_custom.layers[-1].set_weights([custom_weights, custom_biases])

		return network_custom


	def _create_dual_net_PQ( self, network, property ):
		output_size = len(property["Q"])

		output_custom = tf.keras.layers.Dense(output_size*2, activation='linear')(network.output)
		network_custom = tf.keras.Model( network.input, output_custom )

		custom_biases = np.array([[a, -b] for a, b in property["Q"]]).flatten()
		custom_weights = np.zeros((output_size, output_size*2))
		
		for i in range(output_size):
			custom_weights[i][i*2+0] = -1
			custom_weights[i][i*2+1] = 1

		network_custom.layers[-1].set_weights([custom_weights, custom_biases])

		return network_custom


	def _create_dual_net_positive( self, network ):
		output_size = network.output.shape[1]

		output_custom = tf.keras.layers.Dense(output_size, activation='linear')(network.output)
		network_custom = tf.keras.Model( network.input, output_custom )

		custom_biases = np.zeros(output_size)
		custom_weights = np.zeros((output_size, output_size))

		for i in range(output_size):
			custom_weights[i][i] = -1

		network_custom.layers[-1].set_weights([custom_weights, custom_biases])

		return network_custom

