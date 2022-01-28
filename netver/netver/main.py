import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from netver.backend.ProVe import ProVe
from netver.backend.CompelteProve import CompleteProVe
import tensorflow as tf; import numpy as np

class NetVer:

	# property type: [PQ (P, Q), Positive (P), Decision (P, A never selected)]

	algorithms_dictionary = {
		"prove" : ProVe,
		"complete_prove" : CompleteProVe
	}


	def __init__( self, algo, network, property, **kwargs ):

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

		print( "asdasdasda" )

		return self.verifier.verify( verbose )

	
	def estimate_vr( self, test_size=10000 ):
		domains = [ np.random.uniform(node[0], node[1], (test_size, 1) ) for node in self.P ]
		network_input = np.concatenate(domains, axis=1)
		network_output = self.primal_network(network_input).numpy()
		
		counter = 0
		for out in network_output:
			check = [node >= 0 for node in out]
			if not all(check): counter += 1
				
		return np.round(counter / test_size * 100, 3)


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