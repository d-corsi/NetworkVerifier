from netver.backend.ProVe import ProVe
import numpy as np


class CompleteProVe( ProVe ):

	def __init__(self, network, P, **kwargs):
		super().__init__(network, P, **kwargs)
		self.dual_network = kwargs["dual_network"]


	def verify( self, verbose ):

		# Flatten the input domain to aobtain the areas matrix to simplify the splitting
		areas_matrix = np.array([self.P.flatten()])

		# Array with the number of violations for each depth level
		violation_rate_array = []
		
		# Loop until all the subareas are eliminated (verified) or a counter example is found
		for cycle in range(self.time_out_cycle):

			# Print some monitoring information
			if verbose > 0: print( f"Iteration cycle {cycle:3d} of {self.time_out_cycle:3d} (checked  {100-self.unchecked_area:5.3f}%)" )

			# Eventually round the areas matrix
			if self.rounding is not None: areas_matrix = np.round(areas_matrix, self.rounding)
			
			# Reshape the areas matrix in the form (N, input_number, 2)
			test_domain = areas_matrix.reshape(-1, self.P.shape[0], 2)

			# Call the propagation method to obtain the output bound from the input area (primal and dual)
			test_bound = self._propagation_method( test_domain, self.network )
			test_bound_dual = self._propagation_method( test_domain, self.dual_network )

			# Call the verifier (N(x) >= 0) on all the subareas
			unknown_id, violated_id, proved_id = self._complete_verifier( test_bound, test_bound_dual )

			# Call the updater for the checked area
			self._update_unchecked_area( cycle, proved_id[0].shape[0]+violated_id[0].shape[0] )
			
			# Update the violation rate array to compute the violation rate
			violation_rate_array.append( len(violated_id[0]) )

			# Iterate only on the unverified subareas
			areas_matrix = areas_matrix[unknown_id]

			# Exit check when all the subareas are verified
			if areas_matrix.shape[0] == 0: break

			# Exit check when the checked area is below the timout threshold
			if self.unchecked_area < self.time_out_checked:
				# Update the violation rate array adding all the remaining elements
				violation_rate_array[-1] += areas_matrix.shape[0]
				break

			# Split the inputs (Iterative Refinement)
			areas_matrix = self._split( areas_matrix )

		# Compute the violation rate, multipling the depth for the number for each violation
		# and normalizing for the number of theoretical leaf
		violations_weigth = sum( [ 2**i * n for i, n in enumerate(reversed(violation_rate_array))] ) 
		violation_rate =  violations_weigth / 2**(len(violation_rate_array)-1) * 100 

		# All the input are verified, return SAT with no counter example
		return (violation_rate == 0), { "violation_rate": violation_rate }



	def _complete_verifier( self, test_bound, test_bound_dual ):

		if not self.reversed:
			proved_bound = np.all(test_bound[:, :, 0] >= 0, axis=1) # Property proved here!
			violated_bound = np.any(test_bound_dual[:, :, 0] > 0, axis=1) # Property violated here!
		else:
			proved_bound = np.any(test_bound[:, :, 0] > 0, axis=1) # Property proved here!
			violated_bound = np.all(test_bound_dual[:, :, 0] >= 0, axis=1) # Property violated here!

		unknown_mask = np.logical_or(proved_bound, violated_bound)
		
		unknown_id = np.where( unknown_mask == False )
		violated_id = np.where( violated_bound == True )
		proved_id = np.where( proved_bound == True )

		#
		return unknown_id, violated_id, proved_id

