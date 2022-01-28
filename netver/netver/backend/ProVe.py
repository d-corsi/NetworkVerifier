import netver.utils.propagation_utilities as prop_utils
import numpy as np


class ProVe( ):

	# Verification hyper-parameters
	semi_formal = False
	semi_formal_precision = 100
	time_out_cycle = 35
	time_out_checked = 0
	rounding = None
	reversed = False

	
	def __init__( self, network, P, **kwargs ):

		# Input parameters
		self.network = network
		self.P = P

		# Override the default parameters
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		# Private variables
		self.unchecked_area = 100
		self.split_matrix = self._generate_splitmatrix()

		# Propagation method
		if self.semi_formal: self._propagation_method = prop_utils.multi_area_estimator
		else: self._propagation_method = prop_utils.multi_area_propagation_cpu

	
	def verify( self, verbose ):

		# Flatten the input domain to aobtain the areas matrix to simplify the splitting
		areas_matrix = np.array([self.P.flatten()])
				
		# Loop until all the subareas are eliminated (verified) or a counter example is found
		for cycle in range(self.time_out_cycle):

			# Print some monitoring information
			if verbose > 0: print( f"Iteration cycle {cycle:3d} of {self.time_out_cycle:3d} (checked {(100-self.unchecked_area):6.3f}%)" )

			# Eventually round the areas matrix
			if self.rounding is not None: areas_matrix = np.round(areas_matrix, self.rounding)

			# Reshape the areas matrix in the form (N, input_number, 2)
			test_domain = areas_matrix.reshape(-1, self.P.shape[0], 2)

			# Check for a violation in the current domains
			counter_example = self._test_counter_example( test_domain )

			# If there is a counter example, return UNSAT with the example
			if counter_example is not None: return False, { "counter_example": counter_example }

			# Call the propagation method to obtain the output bound from the input area
			test_bound = self._propagation_method( test_domain, self.network )

			# Call the verifier (N(x) >= 0) on all the subareas
			unknown_id, proved_id = self._verify_property( test_bound )

			# Call the updater for the checked area
			self._update_unchecked_area( cycle, proved_id[0].shape[0] )

			# Iterate only on the unverified subareas
			areas_matrix = areas_matrix[unknown_id]

			# Exit check when all the subareas are verified
			if areas_matrix.shape[0] == 0: break

			# Exit check when the checked area is below the timout threshold
			if self.unchecked_area < self.time_out_checked: 
				# return UNSAT with the no example but a
				return False, { "counter_example" : None , "exit_reason" : "exploration_timeout" }

			# Split the inputs (Iterative Refinement)
			areas_matrix = self._split( areas_matrix )

		# All the input are verified, return SAT with no counter example
		return True, { "counter_example": None }

	

	def _test_counter_example( self, test_domain ):

		domains = [ [(node[0] + node[1])/2 for node in area] for area in test_domain ]
		network_input = np.vstack(domains)

		network_output = self.network(network_input).numpy()
		
		if not self.reversed:
			mins = np.min(network_output, axis=1) 
			violation_id = np.where( mins < 0)[0]
		else:
			maxi = np.max(network_output, axis=1) 
			violation_id = np.where( maxi <= 0)[0]
		
		counter_example = None if violation_id.shape[0] == 0 else network_input[violation_id[0]]

		return counter_example


	def _verify_property( self, test_bound ):
		if not self.reversed:
			proved_bound = np.all(test_bound[:, :, 0] >= 0, axis=1) # Property proved here!
		else:
			proved_bound = np.any(test_bound[:, :, 0] > 0, axis=1) # Property proved here!

		unknown_id = np.where(proved_bound == False)
		proved_id = np.where(proved_bound == True)
		return unknown_id, proved_id

	
	def _update_unchecked_area( self, depth, verified_nodes ):
		self.unchecked_area -= 100/ 2**depth * verified_nodes
		

	def _split( self, areas_matrix ):
		i = self._chose_node( areas_matrix )
		res = (areas_matrix.dot(self.split_matrix[i]))
		areas_matrix = res.reshape((len(res) * 2, self.P.shape[0] * 2))
		return areas_matrix


	def _chose_node( self, area_matrix ):
		first_row = area_matrix[0]
		distances = []
		closed = []
		for index, el in enumerate(first_row.reshape(self.P.shape[0], 2)):
			distance = el[1] - el[0]
			if(distance == 0): closed.append(index)
			distances.append(el[1] - el[0])
		return distances.index(max(distances))


	def _generate_splitmatrix( self ):

		rounding_fix = 0 if self.rounding is None else 1/10**(self.rounding+1)
		n = (self.P.shape[0] * 2)
		split_matrix = []
		for i in range( self.P.shape[0] ):
			split_matrix_base = np.concatenate((np.identity(n, dtype="float32"), np.identity(n, dtype="float32")), 1)
			split_matrix_base[(i * 2) + 0][(i * 2) + 1] = 0.5 
			split_matrix_base[(i * 2) + 1][(i * 2) + 1] = (0.5 - rounding_fix)
			split_matrix_base[(i * 2) + 0][n + (i * 2)] = 0.5 
			split_matrix_base[(i * 2) + 1][n + (i * 2)] = (0.5 + rounding_fix)
			split_matrix.append(split_matrix_base)
		return split_matrix

