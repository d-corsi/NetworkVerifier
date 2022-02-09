cuda_code = '''

extern "C" __global__ void my_kernel(float* input_domain, int input_domain_n, int* layer_sizes, int layer_number, float* full_weights, 
			float* full_biases, float* results_cuda, int max_layer_size, int* activations) {

	// Calculate all the bounds, node by node, for each layer. 'new_layer_values' is the current working layer, old layer is the prevoius (first step old layer is the input layer)
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_id >= input_domain_n) return;
	int area_start = thread_id * layer_sizes[0] * 2;
	
	float* old_layer_values = new float[max_layer_size * 2]();
	float* new_layer_values = new float[max_layer_size * 2]();

	// Step 1: copy inputs in 'old_layer_values' ('new_layer_values' is the first hidden layer)
	for (int i = 0; i < (2 * layer_sizes[0]); i++) old_layer_values[i] = input_domain[area_start + i];
	
	// Step 2: starting the propagation cycle
	int bias_index = 0;
	int weights_index = 0;
	for (int layer_idx = 0; layer_idx < layer_number - 1; layer_idx ++){
		int old_layer_size = layer_sizes[layer_idx];
		int new_layer_size = layer_sizes[layer_idx + 1];
		
		for (int new_node_idx = 0; new_node_idx < new_layer_size*2; new_node_idx += 2){
			for (int old_node_idx = 0; old_node_idx < old_layer_size*2; old_node_idx += 2){
				if(full_weights[weights_index] > 0) {
					new_layer_values[new_node_idx] += (old_layer_values[old_node_idx] * full_weights[weights_index]); //lower bound
					new_layer_values[new_node_idx + 1] += (old_layer_values[old_node_idx + 1] * full_weights[weights_index]); //upper bound
				} else {
					new_layer_values[new_node_idx] += (old_layer_values[old_node_idx + 1] * full_weights[weights_index]); //lower bound
					new_layer_values[new_node_idx + 1] += (old_layer_values[old_node_idx] * full_weights[weights_index]); //upper bound
				}
				weights_index += 1;
			}

			// Adding bias for each layer (including the output)
			new_layer_values[new_node_idx] += full_biases[bias_index];
			new_layer_values[new_node_idx+1] += full_biases[bias_index];  
			bias_index += 1;

			// Application of the activation function
			// ReLU
			if (activations[layer_idx] == 1){
				if (new_layer_values[new_node_idx] < 0)  new_layer_values[new_node_idx] = 0;
				if (new_layer_values[new_node_idx+1] < 0)  new_layer_values[new_node_idx+1] = 0;
			// TanH
			} else if (activations[layer_idx] == 2){
				new_layer_values[new_node_idx] = ( 1 - pow(2.71828f, -2*new_layer_values[new_node_idx]) ) / ( 1 + pow(2.71828f, -2*new_layer_values[new_node_idx]) );
				new_layer_values[new_node_idx+1] = ( 1 - pow(2.71828f, -2*new_layer_values[new_node_idx+1]) ) / ( 1 + pow(2.71828f, -2*new_layer_values[new_node_idx+1]) );
			// Sigmoid
			} else if (activations[layer_idx] == 3){
				new_layer_values[new_node_idx] = 1 / ( 1 + pow(2.71828f, -new_layer_values[new_node_idx]) );
				new_layer_values[new_node_idx+1] = 1 / ( 1 + pow(2.71828f, -new_layer_values[new_node_idx+1]) );
			}
		}
		for (int i = 0; i < max_layer_size * 2; i++) old_layer_values[i] = new_layer_values[i];
		for (int i = 0; i < max_layer_size * 2; i++) new_layer_values[i] = 0;
	}

	// Step 3: copy the local output layer in the global 'results_cuda' array
	int results_start = thread_id * layer_sizes[layer_number - 1] * 2;
	for (int i=0; i < layer_sizes[layer_number - 1] * 2; i++) results_cuda[results_start + i] = old_layer_values[i];
	// Free memory
	delete[] old_layer_values;
	delete[] new_layer_values;        
}

'''