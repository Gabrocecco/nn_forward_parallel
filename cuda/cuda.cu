#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>  // for generationg random values in GPU

// nvcc cuda.cu -o cuda
// ./cuda <n_input_neurons> <R> <n_layers>
// ./cuda 1000000 3 100

const float bias = 0.1; // Constant bias 

// Sigmoid function, simple version 
__device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

// Kernel that calculates a sone output values of next output layer 
__global__ void compute_layerGPU(  float *activations,   
                                float *weights,  
                                float next_layer_size,  
                                float R,      
                                int activations_offset,
                                int weights_offset,          
                                int output_idx      
                             ) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // index of the output neuron 
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < next_layer_size; i += stride)
    {
        float sum = 0.0;
        for (int r = 0; r < R; r++) {
            sum += input[activations_offset + idx + r] * weights[weights_offset + idx * R + r];
        }
        activations[idx] = sigmoid(sum + bias);
    }
}

// initalizate an array with random float values (0, 1) range
__global__ void initializeRandomArray(float *array, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);  // Initialize the RNG with a seed
        array[idx] = curand_uniform(&state);  // Generate a random float in [0, 1)
    }
}

int main(int argc, char *argv[]) {

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    printf("Number of SMs: %d\n", numberOfSMs);

    float tstart, tstop;
    long tot_number_of_bytes_allocated = 0;

    if (argc != 4) {
        printf("Usage: %s <N> <R> <K>\n", argv[0]);
        return -1;
    }

    // Read input params 
    int N = atoi(argv[1]);
    int R = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("N=%d, R=%d, K=%d\n", N, R, K);

    // Compute total number of weights 
    int total_weights = 0;
    int layer_size;
    for (int t = 0; t < K - 1; t++) {   // we have weights for K-1 layers (we don't have weights for the input )
        layer_size = N - t * (R - 1);   // numbers of neurons for the current layer 
        total_weights += layer_size * R;    // we are R unique weights for each neuron
    }
    printf("Output layer size: %d\n", layer_size);
    printf("Total number of weigths: %d\n", total_weights);

    // Data allocation on CPU 
    // we want to allocate two large sequential arrays, one for neurons activation 
    // and one for weights 
    int size = sizeof(float);   //both weights and activation are float's

    // printf("CPU allocation...\n");
    tstart = hpc_gettime();
    float *activationsCPU = (float *)malloc(total_neurons * size);
    float *weightsCPU = (float *)malloc(total_weights * size);
    tstop = hpc_gettime();
    
    tot_number_of_bytes_allocated += (K * sizeof(float *)) + (total_weights * sizeof(float));

    // GPU allocation and initialization 
    float *activationsGPU, *weightsGPU;
    cudaMalloc(&activationsGPU, total_neurons * size);
    cudaMalloc(&weightsGPU, total_weights * size);
    
    // Launch kernel to initialize weights with random values
    int seed = 99;
    int threads_per_block = 1024;
    int blocksPerGrid = (total_weights + threadsPerBlock - 1) / threadsPerBlock;
    initializeRandomArray<<<blocksPerGrid, threadsPerBlock>>>(weightsGPU, total_weights, seed);

    // Copy data back from GPU to CPU
    cudaMemcpy(activationsCPU, activationsGPU, total_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weightsCPU, weightsGPU, total_weights * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify by printing fisrt and last 10 values of weights
    for (int i = 0; i < 10; i++) {
        printf("weightsCPU[%d] = %f ", i, weightsCPU[i]);
    } 
    printf("\n");
    for (int i = total_weights - 10; i < total_weights; i++) {
        printf("weightsCPU[%d] = %f ", i, weightsCPU[i]);
    }

    int activations_offset = 0;
    int weights_offset = 0;
    tstart = hpc_gettime();
    for (int t = 1; t < K; t++) {   // we iterate from layer 1 to layer K-1
        int input_layer_size = N - (t-1) * (R - 1);   // input layer size
        int output_layer_size = N - t * (R - 1);  // output layer size

        int output_idx = activations_offset + input_layer_size;    

        // Numero di thread per blocco
        int numBlocks = (output_layer_size + threadsPerBlock - 1) / threadsPerBlock;
        
        // Chiamata al kernel CUDA
        compute_layerGPU<<<numBlocks, threadsPerBlock>>>(activationsGPU, weightsGPU, output_layer_size, R, activations_offset, weights_offset, output_idx);
        cudaDeviceSynchronize(); // Wait for the GPU to finish
    
        // update the activation offset at the first neuron of the next input layer
        activations_offset += input_layer_size;
        weights_offset += output_layer_size * R;
    }
    tstop = hpc_gettime();
    printf("Compute time GPU: %.10f\n", tstop - tstart);

    // Copy data back from GPU to CPU
    cudaMemcpy(activationsCPU, activationsGPU, total_neurons * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify by printing last 10 values of activations
    for (int i = total_neurons - 10; i < total_neurons; i++) {
        printf("activationsCPU[%d] = %f ", i, activationsCPU[i]);
    }

    // Deallocazione memoria
    free(activationsCPU);
    free(weightsCPU);
    
    // Deallocazione memoria sulla GPU
    cudaFree(activationsGPU);
    cudaFree(weightsGPU);
}