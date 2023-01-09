import numpy as np;
from tqdm import tqdm
import operator
import math
import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#from netver.main import NetVer
from functools import reduce

class CountingProVe(  ):

	"""
	A class that implements Counting ProVe, a randomized-approximation method to solve the #DNN-verification problem[a]. 
	This tool provides an approximated solution with formal guarantees on the confidence interval of the input area that has violations. 
	
	[a] Marzari, Corsi et al. The #DNN-Verification problem: Counting Unsafe Inputs for Deep Neural Networks coming soon ....

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

	Methods
	-------
		verify( verbose )
			method that formally verify the property P on the given network
	"""

	# Verification hyper-parameters
	estimated_true_count = False
	time_out_cycle = 40
	time_out_checked = 0.0
	rounding = 4
	reversed = False
	
	# Hyperparameters for confidence at 99%, i.e., β*t=7 and the number of samples to compute the median 
	beta = 0.02
	T = 350
	m = 1500000
	



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

	
		# Input parameters
		self.network = network
		self.P = P
		self.area = np.array(self.P, dtype=np.float64).copy()
		self.initial_area_size = self._compute_area_size()
		self.dual_network = dual_network


		# Override the default parameters
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		# set how many preliminary splits are necessary
		discretization = self.rounding
		r = (1/10**(discretization))
		total_area_points = reduce(operator.mul,[ int(np.round(((el[1]-el[0])/r))+1) for el in self.area])
		self.S = int(math.log(total_area_points))

		print(f'\n\tHyperparameters: [S = {self.S - 1}, β = {self.beta}, T = {self.T}, m = {self.m}, confidence = {round((1 - 2 ** (-self.beta*(self.T)))*100, 2)}%]\n')



	def verify( self, verbose ):

		"""
		Method that perform the analysis.
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


		# MAIN LOOP
		if verbose: print('Start computing the confidence interval of VR:\n')

		# start assuming the max violation and safe rate
		lower_violation_rate = 1
		lower_safe_rate = 1

		for _ in tqdm(range(self.T)):

			self.area = np.array(self.P, dtype=np.float64).copy()
			violation_t = self.count( self.beta, self.S )

			self.area = np.array(self.P, dtype=np.float64).copy()
			safe_t = self.count( self.beta, self.S, count_violation=False )

			lower_violation_rate = min(lower_violation_rate, violation_t)
			lower_safe_rate = min(lower_safe_rate, safe_t)

		lower_bound = round(lower_violation_rate*100, 3)
		upper_bound = round(lower_safe_rate*100, 3)
		violation_rate = round((lower_bound + (100 - upper_bound))/2, 3)


		return (violation_rate == 0), { "lower_bound": lower_bound, "upper_bound": 100 - upper_bound, "size_interval_confidence_VR": (100 - upper_bound)-lower_bound, "violation_rate": violation_rate}



	def count( self, beta, S, count_violation=True):	

		initial_area = self._compute_area_size(self.area)
		node_index = -1
		
		for _s in range(S):

			node_index += 1
			if node_index > (self.area.shape[0]-1): node_index = 0
			_, violated_points, _ = self._get_sampled_violation(  input_area=self.area, cloud_size=self.m, violation=count_violation )			
			
			if violated_points.shape[0] == 0: 
				S = _s
				break
			
			median = np.median( violated_points[:, node_index] )
			random_side = np.random.randint(0, 2)
			self.area[node_index][random_side] = median
	

		# Compute the violation rate of the current leaf with the formal method
		if not self.estimated_true_count:
			prp = { "type" : "positive", "P" : self.area }
			netver = NetVer( "complete_prove", self.network, prp, rounding=self.rounding, time_out_checked=self.time_out_checked )
			_, info = netver.run_verifier( verbose=0)
			rate_split = (info['violation_rate'] / 100)
			if not count_violation: rate_split = 1 - rate_split

		else:
			#netver = NetVer( "estimated", self.network, prp, cloud_size=1000000 )
			rate_split, _, _ = self._get_sampled_violation(  input_area=self.area, cloud_size=1000000, violation=count_violation )

	
		area_leaf = self._compute_area_size(self.area)
		violated_leaf_points = area_leaf * rate_split
		ratio = violated_leaf_points / initial_area
		return 2**(S-beta) * ratio




	#######################
	###	PRIVATE METHODS ###
	#######################


	def _compute_area_size( self, area=None ):

		if area is None: area = self.area 

		area_size = reduce(operator.mul,[(el[1]-el[0]) for el in area])
		return area_size

	
	def _get_sampled_violation( self, cloud_size=1000, input_area=None, violation=True ):

		if input_area is None: input_area = self.area
		network_input = self._generate_input_points( cloud_size, input_area )

		num_sat_points, sat_points = self._get_rate(self.network, network_input, violation)
		rate = (num_sat_points / cloud_size)

		return rate, sat_points, network_input.shape[0]


	def _generate_input_points( self, cloud_size, input_area ):

		input_area = input_area.reshape(1, input_area.shape[0], 2) 
		domains = np.array([np.random.uniform(i[:, 0], i[:, 1], size=(cloud_size, input_area.shape[1])) for i in input_area])
		network_input = domains.reshape( cloud_size*input_area.shape[0], -1 )

		return network_input


	def _get_rate(self, model, network_input, violation):
		
		model_prediction = model(network_input).numpy()

		if violation:
			where_indexes = np.where([model_prediction < 0])[1]
		else:
			where_indexes = np.where([model_prediction >= 0])[1]

		input_conf = network_input[where_indexes]

		return len(where_indexes), input_conf