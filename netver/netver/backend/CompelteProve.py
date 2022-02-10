from netver.backend.ProVe import ProVe
import numpy as np; import tensorflow as tf


class CompleteProVe( ProVe ):

	"""
	A class that implements Complete ProVe, a verification tool based on the interval propagation. 
	This tool is based on a parallel implementation of the interval analysis on GPU that increase the performance. 
	The main difference between ProVe and Complete ProVe is that the second can also formally verify a violation and not only
	when a property is respected. This results allows Complete ProVe to compute the vaiolation rate, the percentage of the input domain 
	that cause a violation, this tool does not terimante as soon as it found a counterexample, but seraches an all the input-domain to prove (or deny) 
	the proeprty at each point bound-wise (see [b] for the details). 
	[a] coming soon ....

	Attributes
	----------
		P : list
			input domain for the property in the form 'positive', each output from a point in this domain must be greater than zero.
			2-dim list: a list of two element (lower_bound, upper_bound) for each input nodes
		network : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
		dual_network: tf.keras.Model
			the dual netowrk is built to deny the properties, is the negation of the main netowrk, a property is violated when "at least
			ONE output of the dual netowrk is greater than zero. It also works in "reverse" mode, a property is violated when 
			"ALL the outputs of the dual netowrk are greater than zero"
		super: super()
			this class is inherited from netver.backend.ProVe, all the paramters of the parent class are inherited in this class

	Methods
	-------
		verify( verbose )
			method that formally verify the property P on the ginve network
	"""

	def __init__(self, network, P, dual_network, **kwargs):

		"""
        Constructor of the class, also calls the super class constructor ProVe

        Parameters
        ----------
			network : tf.keras.Model
				tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
            P : list
				input domain for the property in the form 'positive', each output from a point in this domain must be greater than zero.
				2-dim list: a list of two element (lower_bound, upper_bound) for each input nodes
        """

		super().__init__(network, P, **kwargs)
		self.dual_network = dual_network


	def verify( self, verbose ):

		"""
        Method that perform the formal analysis. When the solver explored and verify all the input domain it returns the
		violation rate, when the violation rate is zero we colcude the the proeprty is fully respected, i.e., SAT

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
				key 'violation_rate' returns the value of the vilation rate as a percentage of the input domain
        """

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

		# Check if the exit reason is the time out on the cycle
		if cycle >= self.time_out_cycle:
			# return UNSAT with the no counter example, specifying the exit reason
			return False, { "counter_example" : None, "exit_code" : "cycle_timeout" }

		# Compute the violation rate, multipling the depth for the number for each violation
		# and normalizing for the number of theoretical leaf
		violations_weigth = sum( [ 2**i * n for i, n in enumerate(reversed(violation_rate_array))] ) 
		violation_rate =  violations_weigth / 2**(len(violation_rate_array)-1) * 100 

		# All the input are verified, return SAT with no counter example
		return (violation_rate == 0), { "violation_rate": violation_rate }



	def _complete_verifier( self, test_bound, test_bound_dual ):
		
		"""
        Method that verify the property on a list of the computed (or sampled in semi-formal mode) output bound.

        Parameters
        ----------
            test_bound : list
                the output bound expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
				(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
			test_bound_dual : list
                same as test_bound but for the dual network

		Returns:
		--------
			unknown_id : list
				list of integer with the index of the bound that dows not respect the property and require
				further investigations
			violated_id : list
				list of integer with the index of the bound that violated the give property
			proved_id : list
				list of integer with the index of the bound that respect the give property
        """

		# Check the property in standard and reverse mode for both a violation and a proof. 
		# To prove the property, in the first case every bound must be greater than zero,
		# in the latter at least one bound must be greater than zero. 
		# To deny the property, in the first case at least one bound must be greater than zero,
		# in the latter every one bound must be greater than zero. 
		if not self.reversed:
			proved_bound = np.all(test_bound[:, :, 0] >= 0, axis=1) # Property proved here!
			violated_bound = np.any(test_bound_dual[:, :, 0] > 0, axis=1) # Property violated here!
		else:
			proved_bound = np.any(test_bound[:, :, 0] > 0, axis=1) # Property proved here!
			violated_bound = np.all(test_bound_dual[:, :, 0] >= 0, axis=1) # Property violated here!

		# Create a mask for the unknown, when a property is neither proved or vioalted
		unknown_mask = np.logical_or(proved_bound, violated_bound)

		# Find the unknown and proved index with the built-in numpy function		
		unknown_id = np.where( unknown_mask == False )
		violated_id = np.where( violated_bound == True )
		proved_id = np.where( proved_bound == True )

		#
		return unknown_id, violated_id, proved_id

