from itertools import count
import numpy as np
import tensorflow as tf
from z3 import *

class LinearSolver:

	"""
	A class that implements a general version of the solver based on the linearization of the problem. 
	The cited paper are enhanced version of the implemented one. The version in this repo is the vanilla version, 
	look at the original paper to understand the optimized versions:	
	[a] Dutta et al., Output Range Analysis for Deep Neural Networks, NASA Formal Methods Symposium, 2018
	[b] Lomuscio et al., An Approach to Reachability Analysis for Feed Forward ReLU Neural Networks, adsabs.harvard, 2017
	[c] Ehlers et al., Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks, Automated Technology for Verification and Analysis, 2017

	Attributes
	----------
		P : list
			input domain for the property in the form 'positive', each output from a point in this domain must be greater than zero.
			2-dim list: a list of two element (lower_bound, upper_bound) for each input nodes
		network : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
		solver: z3.Solver
			the z3 low level solver for the SMT problem
		reversed: bool
			this variables represent that the verification query is reversed, it means that "at least ONE output must be greater than 0" instead of the common
			form where "ALL inputs must be greater than zero".
		*_variables: list
			list of the variables for the solver, two for each node, pre and post activation

	Methods
	-------
		verify( verbose )
			method that formally verify the property P on the ginve network
	"""


	# Verification hyper-parameters
	reversed = False


	def __init__( self, network, P, **kwargs ):

		"""
		Constructor of the class.

		Parameters
		----------
			network : tf.keras.Model
				tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
			P : list
				input domain for the property in the form 'positive', each output from a point in this domain must be greater than zero.
				2-dim list: a list of two element (lower_bound, upper_bound) for each input nodes
			kwargs : **kwargs
				dicitoanry to overide all the non-necessary paramters (if not specified the algorithm will use the default values)	
		"""

		# Input parameters
		self.P = P
		self.network = network
		
		# Override the default parameters
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		# Load the weights
		self.weights = [ layer.get_weights()[0].T for layer in network.layers[1:] ]
		self.biases = [ layer.get_weights()[1] for layer in network.layers[1:] ]

		# Create the variables for each node (input, pre-activation-hidden, post-activation-hidden)
		self.input_variables = np.array([ Real(f"x{i}") for i in range(network.input.shape[1]) ])
		self.hidden_variables = np.array([[ Real(f"h_{i}_{j}") for j in range(0, len(self.weights[i]))] for i in range(0, len(self.weights)) ][:-1], dtype=object)
		self.activated_variables = np.array([[ Real(f"a_{i}_{j}") for j in range(0, len(self.weights[i]))] for i in range(0, len(self.weights)) ][:-1], dtype=object)
		self.output_variables = np.array([ Real(f"y{i}") for i in range(network.output.shape[1]) ])
		self.output_variables_activated = np.array([ Real(f"r{i}") for i in range(network.output.shape[1]) ])

		# Count the number of relu variables in the hidden layers
		self.relu_variables = 0
		for j, _ in enumerate(self.activated_variables):
			if self._is_relu(j): self.relu_variables += len(self.activated_variables[j])

		# Count the number of relu variables in the output layer
		if self._is_relu(j+1): self.relu_variables += len(self.output_variables)

		# Initialization of the Linear solver (z3)
		self.solver = Solver()

		# Add constraints on hidden layers
		for j, _ in enumerate(self.hidden_variables):
			if j == 0:
				for i, h in enumerate(self._linear_combination(self.input_variables, self.weights[0], self.biases[0])): 
					self.solver.add( self.hidden_variables[0][i] == h )
			if j != 0:
				for i, h in enumerate(self._linear_combination(self.activated_variables[j-1], self.weights[j], self.biases[j])): 
					self.solver.add( self.hidden_variables[j][i] == h )	

		# Add constraints on output layer
		self.solver.add( [self.output_variables[i] == h for i, h in enumerate(self._linear_combination(self.activated_variables[-1], self.weights[-1], self.biases[-1]))] )

		# Add constraints on input/output
		self.solver.add( [self.input_variables[i] >= l for i, l in enumerate(self.P[:, 0])] )
		self.solver.add( [self.input_variables[i] <= u for i, u in enumerate(self.P[:, 1])] )

		# positive -> (SAT if each output have a value >= 0)
		# reverse positive -> (SAT if at least one output have value > 0)
		# Add constraints on the output nodes, looking to falsify the property 
		if not self.reversed: self.solver.add( z3.Or([output_var < 0 for output_var in self.output_variables_activated]) )
		else: self.solver.add( [output_var < 0 for output_var in self.output_variables_activated] )


	def verify( self, verbose ):

		"""
		Method that perform the formal analysis.
		When the z3 solver returns SAT it means that a counterexample is found; ergo the global formal verification 
		return UNSAT beacuse there is an input configuration that leads to a violation of the property P.
		The method create all the possible combination of ReLU in the two possible lienar phase (2**relu_node) and then analyze them
		as a standard linear program. If one of the leaf is SAT a counterexample is found, to prove that no coutnerexample exists it 
		is necessary to analyze all the leaf.

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
		"""

		# Compute the number of leaf, this represents the maximum number of problem to analyze
		leaf_number = 2 ** self.relu_variables

		# Iterate over (in the worst case) all the leaf (i.e., possible linear configuration)
		for cycle in range(leaf_number):

			# If necessary print some log informations
			if verbose > 0: print( f"Worst case iterations: {cycle}/{leaf_number}" )

			# Generate the configuration for the relu activations function, a configuration
			# is a binary string with 0 if the phase of the corresponding node is inactive
			# and 1 if active
			conf = "{0:b}".format(cycle)
			for _ in range(self.relu_variables-len(conf)): conf = "0" + conf

			# Check for the current linear problem (z3 instance)
			sat = self._verify_linear_configuration( conf )

			# If the problem is SAT, a counterexample is found
			if sat: 

				# Compute the counterexample checkong the state of the 
				# input variables of the SAT results and return
				counter_example = [  self._z3_real_to_float(var) for var in self.input_variables ]
				return False, { "counter_example": counter_example }

			# Restoring the state before the scope
			self.solver.pop()

		# All the input are verified, return SAT with no counter example
		return True, { "counter_example": None }

	
	def _verify_linear_configuration( self, configuration ):

		"""
		Method that solve a linear problem with the backend solver z3. To transform the non linear problem
		of the given network, this method set each hidden ReLU node to a fixed linear phase according to
		the given configuration.

		Parameters
		----------
			configuration : str
				a binary string with 0 if the phase of the corresponding node is inactive and 1 if active

		Returns:
		--------
			sat : bool
				Return the result of the SAT analysis with z3. If the result is UNSAT no counterexample is found and the 
				proeprty is respected, if the result is SAT it means that we have a counterexample so the proeprty is violated, 
				in this last case even the global verification query stop
		"""

		# Create a new scope
		self.solver.push()

		# Counter for the analyzed ReLU node
		counter = 0
		
		# Decode the input configuration to create the linear program
		for j, _ in enumerate(self.hidden_variables):
			for i, _ in enumerate(self.hidden_variables[j]):

				# This layer is ReLU
				if self._is_relu(j):

					# ReLU Layer
					if(configuration[counter] == '0'):
						# Relu in inactive phase
						self.solver.add( self.hidden_variables[j][i] < 0 ) 
						self.solver.add( self.activated_variables[j][i] == 0 ) 
					if(configuration[counter] == '1'):
						# Relu in active phase
						self.solver.add( self.hidden_variables[j][i] >= 0 ) 
						self.solver.add( self.activated_variables[j][i] == self.hidden_variables[j][i] ) 

					# Update the counter
					counter += 1

				# This layer is not ReLU
				else:
					self.solver.add( self.activated_variables[j][i] == self.hidden_variables[j][i] ) 

				
		# Special case with relu on the last layer
		if (len(configuration)-counter) > 0:
			#
			for output_var in self.output_variables:
				# Relu in inactive phase
				if(configuration[counter] == '0'): self.solver.add( output_var < 0 ) 
				# Relu in active phase
				if(configuration[counter] == '1'): self.solver.add( output_var >= 0 ) 

		# Common case with linear on the last layer
		else:
			for pre, activated in zip(self.output_variables, self.output_variables_activated):
				self.solver.add( activated == pre )

		# Check the current linear problem and return
		return self.solver.check() == z3.sat


	def _z3_real_to_float( self, variable ):

		"""
		Method that convert the z3 variable to a float, useful to obtain
		the value before the return of the counterexample.

		Parameters
		----------
			variable : z3.var
				the variable to convert in string

		Returns:
		--------
			val : float
				the float that represent the z3 variales
		"""

		# Ask to the solver for the real value of interested variable
		real = self.solver.model()[variable]

		# Converting to a fraction and, dividing numerator and denominator, obtaining the float value
		val = real.as_fraction()
		val = float(val.numerator) / float(val.denominator)

		#
		return val
		

	def _linear_combination( self, input_nodes, params, biases ):
		
		"""
		Method that encode the linear combination of each node (layer-wise) of the fully connected network
		with the previous layer, applying the biases if necessary.

		Parameters
		----------
			input_nodes : list
				node in input to the interested layer (previous layer)
			params : list
				parameters of the network, the weights of the layer
			biases : list
				list of the biases in the interested layer to add before the return

		Returns:
		--------
			output : list
				a list with the linear combination formula for each node of the layer
		"""
		
		# Loop to compute the linear combination
		output = []    
		for i, param in enumerate(params):
			node_val = sum([ self._z3_to_string(p) * x for p, x in list(zip(param, input_nodes)) ])
			node_val += biases[i]
			output.append(node_val)
			
		#
		return output



	def _z3_to_string( self, variable ):

		"""
		Method that convert the z3 variable in a string, useful to obtain
		arbitrary decimal precision.

		Parameters
		----------
			variable : z3.var
				the variable to convert in string

		Returns:
		--------
			string : str
				the string that represent the z3 variales with a rpecision of 20 decimals
		"""
		
		#
		return f"{variable:.20f}"


	def _is_relu( self, layer ):

		"""
		Method that return TRUE if the requested layer is ReLU, 
		and FALSE if linear.

		Parameters
		----------
			layer : int
				index of the interested layer

		Returns:
		--------
			is_relu : bool
				true if the ReLu, false otherwise
		"""

		#
		return self.network.layers[layer+1].activation == tf.keras.activations.relu

