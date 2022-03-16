import netver.utils.propagation_utilities as prop_utils
import numpy as np


class Estimated( ):

	"""
	A class that implements an estimator for the real value of the violation rate. The approach is based on a sampling and propagation method, sampling a
	points cloud from the domain of the property the method compute an estimation of the violation rate.
	Givevn a set of point sampled from the domain 'L', the neural network 'N', the propagated set of output 'Y = N(L)', we quantify the number of 
	points in Y that violate the property (i.e., y \in Y < 0).

	Attributes
	----------
		P : list
			input domain for the property in the form 'positive', each output from a point in this domain must be greater than zero.
			2-dim list: a list of two element (lower_bound, upper_bound) for each input nodes
		network : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
		unchecked_area: float
			indicates the percentage of the residual input domain to explore
		cloud_size: int
			indicates the size of the point cloud for the method (default: 10000)
		reversed: bool
			this variables represent that the verification query is reversed, it means that "at least ONE output must be greater than 0" instead of the common
			form where "ALL inputs must be greater than zero".


	Methods
	-------
		verify( verbose )
			method that formally verify the property P on the ginve network
	"""


	# Verification hyper-parameters
	cloud_size = 10000
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
		self.network = network
		self.P = P

		# Override the default parameters
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

	
	def verify( self, verbose ):

		"""
		Method that perform the formal analysis.
		When the solver explored and verify all the input domain the problem is SAT. At each iteration the tool searches for a 
		counter example. If the algorithm find a counterexample it will be return inside within the UNSAT result.

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

		# If necessary print some log informations
		if verbose > 0: print( f"Estimated Verifier with a point cloud of size {self.cloud_size}..." )

		self.P = self.P.reshape(1, self.P.shape[0], 2) 

		# Sampling the point cloud from the input domain and reshaped according to the tensorflow format
		domains = np.array([np.random.uniform(input_area[:, 0], input_area[:, 1], size=(self.cloud_size, self.P.shape[1])) for input_area in self.P])
		network_input = domains.reshape( self.cloud_size*self.P.shape[0], -1 )

		# Propagation of the input through the network
		network_output = self.network(network_input).numpy()

		# Compute the violation rate as a number of points in the point
		# cloud that violates the property (normalized on the size of the point cloud)
		violations = self._enumerate_violation( network_output )
		violation_rate = (violations / self.cloud_size) * 100
		
		# Return UNSAT with the no counter example, specifying the exit reason
		return (violation_rate == 0), { "violation_rate": violation_rate }


	
	def _enumerate_violation( self, output_point_cloud ):

			"""
			Method that search for a counter example in the given domain with a sampling procedure

			Parameters
			----------
				output_point_cloud : list
					the generated set of output from the point cloud (i.e., Y=N(L)) as a 2-dim matrix. 
					(a) a list of list for point of the cloud;
					(b) a list with the value for each output node of the network

			Returns:
			--------
				violations : int
					the number of point in the given point cloud that violate the property
			"""

			print( output_point_cloud.shape)
			
			# Seearch for a violation in standard and reverse mode, in the first case the property is violated if at least one input is lower
			# than zero, in the second case if all the inputs are lower than zero
			if not self.reversed:
				mins = np.min(output_point_cloud, axis=1) 
				violation_id = np.where( mins < 0)[0]
			else:
				maxi = np.max(output_point_cloud, axis=1) 
				violation_id = np.where( maxi <= 0)[0]
			
			# Compute the number of violations in the given point cloud
			violations = len(violation_id) 

			#
			return violations
