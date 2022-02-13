import numpy as np
import tensorflow as tf
from z3 import *

class MILP:

	"""
	A class that implements the MILP verifier:	
	[a] Tjeng et al., Evaluating Robustness of Neural Networks with Mixed Integer Programming, International Conference on Learning Representations, 2018.

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
		self.milp_variables = []

		# Initialization of the MILP solver (z3)
		self.solver = Solver()

		# Add constraints on hidden layers
		for j, _ in enumerate(self.hidden_variables):
			if j == 0:
				for i, h in enumerate(self._linear_combination(self.input_variables, self.weights[0], self.biases[0])): 
					self.solver.add( self.hidden_variables[0][i] == h )
			if j != 0:
				for i, h in enumerate(self._linear_combination(self.activated_variables[j-1], self.weights[j], self.biases[j])): 
					self.solver.add( self.hidden_variables[j][i] == h )	

		# Add constraints on the activation functions layers
		self._parse_hidden_activation()

		# Add constraints on output layer
		self.solver.add( [self.output_variables[i] == h for i, h in enumerate(self._linear_combination(self.activated_variables[-1], self.weights[-1], self.biases[-1]))] )

		# Add constraints on the last layer (activated output)
		milp_counter = 0
		for pre, activated in zip(self.output_variables, self.output_variables_activated):
			if not self._is_relu(j+1):
				# For Linear Activation
				self.solver.add( activated == pre )
			else:
				# For Linear Activation
				print("heeyyy")
				N = 100
				self.milp_variables.append( Int(f"Ty_{milp_counter}") )
				self.solver.add( self.milp_variables[-1] >= 0, self.milp_variables[-1] <= 1)
				self.solver.add( N * self.milp_variables[-1] >= pre)
				self.solver.add( N * (1-self.milp_variables[-1]) >= -pre )
				self.solver.add( activated == (self.milp_variables[-1]*pre) )
				milp_counter += 1

		# Add constraints on the input domain
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

		# If necessary print some log informations
		if verbose > 0: print( "MILP Verifier, launching the z3 verification tool..." )

		# Perform the SAT analysis with z3. 
		# If the result is UNSAT no counterexample is found and the proeprty is respected
		# If the result is SAT it means that we have a counterexample so the proeprty is violated
		if self.solver.check() == z3.unsat: return True, { "counter_example": None }

		# Compute the counterexample checkong the state of the input variables of the SAT results
		counter_example = [  self._z3_real_to_float(var) for var in self.input_variables ]

		#
		return False, { "counter_example": counter_example } 


	def _parse_hidden_activation( self ):

		"""
		Method that perform the analysis of the variables corresponding to the hidden nodes. If a leyer is linear the constraints is only 
		the equality between pre and post activation, if a layer is ReLU è add the integer trick for the MILP verification.

		"""

		milp_counter = 0
		for j, varaible_layer in enumerate(self.hidden_variables):
			for i, _ in enumerate(varaible_layer):		
				if not self._is_relu(j):
					# For Linear Activation only equality
					self.solver.add( self.activated_variables[j][i] == self.hidden_variables[j][i] ) 
				else:
					# For ReLU Activation use the integer trick
					N = 100
					self.milp_variables.append( Int(f"T_{milp_counter}") )
					self.solver.add( self.milp_variables[-1] >= 0, self.milp_variables[-1] <= 1)
					self.solver.add( N * self.milp_variables[-1] >= self.hidden_variables[j][i] )
					self.solver.add( N * (1-self.milp_variables[-1]) >= -self.hidden_variables[j][i] )
					self.solver.add( self.activated_variables[j][i] == (self.milp_variables[-1]*self.hidden_variables[j][i]) )
					milp_counter += 1


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


	def _print_readable_variables( self, only_output=False ):

		"""
		Debug function to print all the variable in a human-readable form.

		Parameters
		----------
			only_output : bool
				flag variable, if true the method only print the output variables, 
				hiding the inner variables (default: False)
		"""
		
		if only_output: 
			flatten_vars = list(self.output_variables) 
		else: 
			flatten_vars = list(self.input_variables.flatten()) 
			hidden = [ [h, a] for h, a in zip(self.hidden_variables.flatten(), self.activated_variables.flatten()) ]
			flatten_vars += sum(hidden, [])
			flatten_vars += list(self.output_variables.flatten()) 	

		for var in flatten_vars:
			val = self.solver.model()[var].as_fraction()
			val = float(val.numerator) / float(val.denominator)
			print( f"{var} = {val}" )

		if not only_output:
			for var in self.milp_variables:
				val = self.solver.model()[var]
				print( f"{var} = {val}" )

