/****************************************************************************
 * Compile with:
 * nvcc cuda.cu -o cuda
 *
 * Run with:
 * ./cuda <n_input_neurons> <R> <n_layers>
 * Use example:
 * ./cuda 10000000 3 10
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>  // for generationg random values in GPU
#include "hpc.h"

const float BIAS = 0.1; // Constant bias 

// Sigmoid function, simple version 
__device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

// Kernel that calculates a single output value one neuron of the next output layer 
__global__ void compute_layerGPU(   float *activations,   // activations array pointer
                                    float *weights,     // weights array pointer
                                    int R,      // number of weights for each neuron 
                                    unsigned long int activations_offset, // index of the first input neuron in the activations array context
                                    unsigned long int weights_offset, // index of the first weightof the first neuron in the weights array context
                                    unsigned long int output_offset   // index of the first output neuron in the activations array context
                                ) 
{
    unsigned long int i = blockIdx.x * blockDim.x + threadIdx.x;    // index of the output neuron
    float sum = 0.0;
    for (int r = 0; r < R; r++) {
        sum += activations[activations_offset + i + r] * weights[weights_offset + (i * R) + r];
    }
    activations[output_offset + i] = sigmoid(sum + BIAS);
}

// initalizate an array with random float values (0, 1) range
__global__ void initializeRandomArray(float *array, unsigned long int size, unsigned long seed) {
    unsigned long int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    cudaEvent_t start, stop;
    float elapsedTime;

    printf("Number of SMs: %d\n", numberOfSMs);

    if (argc != 4) {
        printf("Usage: %s <N> <R> <K>\n", argv[0]);
        return -1;
    }

    // Read input params 
    unsigned long int N = atoi(argv[1]);
    int R = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("N=%lu, R=%d, K=%d\n", N, R, K);

    printf("Data preparation started...\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Compute total number of weights 
    unsigned long int total_weights = 0;
    unsigned long int layer_size;
    unsigned long int total_neurons = N;   // input layer has N neurons 
    for (long t = 1; t < K ; t++) {   // we have weights for K-1 layers (we don't have weights for the input )
        layer_size = N - t * (R - 1);   // numbers of neurons for the current layer 
        total_neurons += layer_size; // update the number of total neurons
        total_weights += layer_size * R;    // we are R unique weights for each neuron
    }
    // printf("Output layer size: %lu\n", layer_size);
    printf("Total number of weigths: %lu\n", total_weights);

    // Data allocation on CPU 
    // we want to allocate two large sequential arrays, one for neurons activation 
    // and one for weights 
    int size = sizeof(float);   //both weights and activation are float's
    
    unsigned long int tot_number_of_bytes_allocated = (total_neurons + total_weights) * size;
    printf("Weigths [GBs]: %.5f\n", (float)tot_number_of_bytes_allocated / 1000000000);

    // GPU allocation and initialization 
    float *activationsGPU, *weightsGPU;
    __cudaCheckError(cudaMalloc(&activationsGPU, total_neurons * size));
    __cudaCheckError(cudaMalloc(&weightsGPU, total_weights * size));
    
    // Launch kernel to initialize weights with random values
    int seed = 99;
    int threadsPerBlock = 512;  // 256/512 best
    int blocksPerGrid = (total_weights + threadsPerBlock - 1) / threadsPerBlock;

    // random initialization of input layers and weights
    initializeRandomArray<<<blocksPerGrid, threadsPerBlock>>>(weightsGPU, total_weights, seed);
    __cudaCheckError( cudaPeekAtLastError() );
    __cudaCheckError( cudaDeviceSynchronize() );
    initializeRandomArray<<<blocksPerGrid, threadsPerBlock>>>(activationsGPU, N, seed);
    __cudaCheckError( cudaPeekAtLastError() );
    __cudaCheckError( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Data preparation time: %f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned long int activations_offset = 0;
    unsigned long int weights_offset = 0;

    printf("Computation starded...\n");
    cudaEventRecord(start, 0);
    for (unsigned long int t = 1; t < K; t++) {   // we iterate from layer 1 to layer K-1
        unsigned long int input_layer_size = N - (t-1) * (R - 1);   // input layer size
        unsigned long int output_layer_size = N - t * (R - 1);  // output layer size

        unsigned long int output_idx = activations_offset + input_layer_size;    

        // Numero di thread per blocco
        unsigned long int numBlocks = (output_layer_size + threadsPerBlock - 1) / threadsPerBlock;
        // printf("Lunching:\n %d blocks of %d threads each. \n Toatal: %d\n", numBlocks, threadsPerBlock, numBlocks * threadsPerBlock);
        // Cuda kernel call
        compute_layerGPU<<<numBlocks, threadsPerBlock>>>(activationsGPU, weightsGPU, R, activations_offset, weights_offset, output_idx);
        __cudaCheckError( cudaPeekAtLastError() );
        __cudaCheckError( cudaDeviceSynchronize() );
    
        // update the activation offset at the first neuron of the next input layer
        activations_offset += input_layer_size;
        weights_offset += output_layer_size * R;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Compute time: %f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copia solo l'ultimo layer sulla CPU
    float *final_layer_activations = (float *)malloc(layer_size * sizeof(float));
    __cudaCheckError(cudaMemcpy(final_layer_activations, 
                                &activationsGPU[activations_offset], 
                                layer_size * sizeof(float), 
                                cudaMemcpyDeviceToHost));

    for (int i = layer_size - 5; i < layer_size; i++) {
        printf("final_layer_activations[%d] = %f\n", i, final_layer_activations[i]);
    }

    // CPU mem deallocation 
    free(final_layer_activations);
    
    // GPU mem deallocation
    cudaFree(activationsGPU);
    cudaFree(weightsGPU);
}
