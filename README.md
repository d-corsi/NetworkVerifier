### Disclaimer:
This is a **work in progress repo**, there may be bugs in the code and numerous typos in the README file. The documentation for the methods and the class structure is a work in progress. For numerous classes the code will be refactored.

# Network Verifier
A set of algorithms for the formal verification and analysis of Neural Networks, implemented in Python for TensorFlow 2. This repo replace the original repo of ProVe (UAI'21), you can find the original repo [here!](https://github.com/d-corsi/ProVe)

## Available Algorithms
- [x] MILP [1]
- [x] Linear Programming Based [2, 3, 4] 
- [ ] Reluplex [5]
- [ ] Reluval [6]
- [ ] Marabou [7]
- [x] ProVe [8]
- [ ] α,β-CROWN [9]
- [x] Complete ProVe [10]

*NB: given the limitations of the original algorithms, [1, 2, 3, 4, 5, 7] are compatible only with piecewise activation functions (e.g., linear, ReLU) while [6, 8, 9, 10] work with all monotonically increasing function (e.g., linear, ReLU, tanh, sigmoid). For [2, 3, 4] we only implement the basic version of the algorithm, please look at the original paper for all the optimizations.*

## Installation

To install the library clone the repo and then use 
```
git clone https://github.com/d-corsi/NetworkVerifier.git
cd NetworkVerifier
pip install -e netver
```

## Definition of the properties
Properties can be defined with 3 different formulations:

### PQ
Following the definition of Marabou [Katz et al.], given an input property P and an output property Q, the property is verified if for each *x* in P it follows that N(x) is in Q *(i.e., if the input belongs to the interval P, the output must belongs to the interval Q)*.
```
property = {
	"type" : "PQ",
	"P" : [[0.1, 0.34531], [0.7, 1.1]],
	"Q" : [[0.0, 0.2], [0.0, 0.2]]
}
```

### Decision
Following the definition of ProVe [Corsi et al.], given an input property P and an output node A corresponding to an action, the property is verified if for each *x* in P it follows that the action A will never be selected *(i.e., if the input belongs to the interval P, the output of node A is never the one with the highest value)*.
```
property = {
	"type" : "decision",
	"P" : [[0.1, 0.3], [0.7, 1.1]],
	"A" : 1
}
```

### Positive
Following the definition of α,β-CROWN [Wang et al.], given an input property P the output of the network is non negative *(i.e., if the input belongs to the interval P, the output of each node is greater or equals zero)*
```
property = {
	"type" : "positive",
	"P" : [[0.1, 0.], [0.7, 1.1]]
}
```


## Compatible Network Models
- [x] Tensorflow 2 *(tf.keras.Model)*
- [ ] Tensorflow 2 *(tf.keras.Sequential)*
- [ ] PyTorch 
- [ ] NNet *(only fully connected ReLU networks)*
- [ ] ONNX  

*example of compatible TF2 model...*
```
inputs = tf.keras.layers.Input(shape=(2,))
hidden_0 = tf.keras.layers.Dense(32, activation='relu', bias_initializer='random_normal')(inputs)
hidden_1 = tf.keras.layers.Dense(32, activation='relu', bias_initializer='random_normal')(hidden_0)
outputs = tf.keras.layers.Dense(5, activation='linear')(hidden_1)

return tf.keras.Model(inputs, outputs)
```


## Run the Algorithm
To use our algorithms use the class **NetVer** from the python file *netver/main.py*. 
```
algorithm_key = "prove" # for ProVe [8]

from netver.main import NetVer
netver = NetVer( algorithm_key, model, property )
sat, info = netver.run_verifier( verbose=1 )
```

Following a lisT of all the algorithms' keyword *(algorithm_key = )*:
```
MILP #[1]
linear #[2, 3, 4]
prove #[8]
complete_prove #[10]
```


## Prameters
NetVer will use the default parameters for the formal analysis. You can change all the parameter when create the NetVer object as follow: 
```
from netver.main import NetVer
netver = NetVer( algorithm_key, model, semi_formal=True, rounding=3 )
```
Follow a list of the available parameters (with the default value):
```
# Common to all the algorithms
time_out_cycle = 35 #timeout on the number of cycle
time_out_checked = 0 #timeout on the checked area, if the unproved area is less than this value the algorithm stop returning the residual as a violation
rounding = None #rounding value for the input domain (P))

# Only for ProVe
semi_formal_precision = 100 #number of samples for the semi formal analysis for each sub interval of the input domain (P)
semi_formal = False #enable the semi-formal verification
```


## Results of the analysis
The analysis returns two values SAT and info. *SAT* is true if the property is respected, false otherwise; *value* is a dictionary that contains different values, based on the used algorithm:

- counter_example: a counter example that falsify the property 
- violation_rate: the violation rate of the property *(only for Complete Prove - complete_prove)*
- exit_reason: reason for an anticipate exit *(usually timeout)*


## Example Code
To run the example code, use *example.py* from the main folder:
```
import tensorflow as tf
import numpy as np
from netver.main import NetVer

def get_actor_model( ):
	inputs = tf.keras.layers.Input(shape=(2,))
	hidden_0 = tf.keras.layers.Dense(32, activation='relu', bias_initializer='random_normal')(inputs)
	hidden_1 = tf.keras.layers.Dense(32, activation='relu', bias_initializer='random_normal')(hidden_0)
	outputs = tf.keras.layers.Dense(5, activation='linear')(hidden_1)

	return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
	print( "Hello World Network Verifier! \n")

	model = get_actor_model()

	property = {
		"type" : "decision",
		"P" : [[0.1, 0.3], [0.7, 1.1]],
		"A" : 1
	}

	netver = NetVer( "complete_prove", model, property, semi_formal=True )

	sat, info = netver.run_verifier( verbose=1 )
	print( f"\nThe property is SAT? {sat}" )
	print( f"\tviolation rate: {info['violation_rate']}\n" )
```


## Author
* **Davide Corsi** - davide.corsi@univr.it


## References
- [1] Tjeng et al., Evaluating Robustness of Neural Networks with Mixed Integer Programming, *International Conference on Learning Representations*, 2018
- [2] Dutta et al., Output Range Analysis for Deep Neural Networks, *NASA Formal Methods Symposium*, 2018
- [3] Lomuscio et al., An Approach to Reachability Analysis for Feed Forward ReLU Neural Networks, *adsabs.harvard*, 2017
- [4] Ehlers et al., Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks, *Automated Technology for Verification and Analysis*, 2017
- [5] Katz et al., Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks, *Computer Aided Verification*, 2017
- [6] Wang et al., Formal Security Analysis of Neural Networks Using Symbolic Intervals, *USENIX Security Symposium*, 2018
- [7] Katz et al., The Marabou Framework for Verification and Analysis of Deep Neural Networks, *International Conference on Computer Aided Verification*, 2019
- [8] Corsi et al., Formal Verification of Neural Networks for Safety-Critical Tasks in Deep Reinforcement Learning, *Conference on Uncertainty in Artificial Intelligence*, 2021
- [9] Wang et al., Efficient Bound Propagation with per-neuron Split Constraints for Complete and Incomplete Neural Network Verification, *Advances in Neural Information Processing Systems*, 2021
- [10] *coming soon...*


## License
- **MIT license**
- Copyright 2021 © **Davide Corsi**.