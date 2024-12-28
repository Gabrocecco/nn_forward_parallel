#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// nvcc cuda.cu -o cuda
// ./cuda <n_input_neurons> <R> <n_layers>
// ./cuda 1000000 3 100

const float bias = 0.1; // Constant bias 

// Sigmoid function, simple version 
__device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

// Kernel that calculates a sone output values of next output layer 
__global__ void compute_layer(  float *input,   // input layer 
                                float *output,  // output layer 
                                float *weights, // array of all weights 
                                float bias,     // constant bias for the layer 
                                int output_size,// nummber of output's neurons 
                                int R,          // constant R 
                                int offset      // offset for weight index 
                             ) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // index of the output neuron 
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < output_size; i += stride)
    {
        float sum = 0.0;
        for (int r = 0; r < R; r++) {
            sum += input[idx + r] * weights[offset + idx * R + r];
        }
        output[idx] = sigmoid(sum + bias);
    }
}

int main(int argc, char *argv[]) {

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

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
    float **layers = (float **)malloc(K * sizeof(float *)); // we use K array's for value activations of the K layers
    float *weights = (float *)malloc(total_weights * sizeof(float));    // we use a single long array for weights
    
    tot_number_of_bytes_allocated += (K * sizeof(float *)) + (total_weights * sizeof(float));
    
    // Allocation for each activation layer on CPU
    for (int t = 0; t < K; t++) {
        int layer_size = N - t * (R - 1);
        layers[t] = (float *)malloc(layer_size * sizeof(float));
        tot_number_of_bytes_allocated += (layer_size * sizeof(float));
        if (t == 0) {  // we initialize at random value only the input layer 
            for (int i = 0; i < layer_size; i++) {
                layers[0][i] = ((float)rand() / RAND_MAX);
            }
        }
    }
    
    // Random weights initialization on CPU
    for (int i = 0; i < total_weights; i++) {
        weights[i] = ((float)rand() / RAND_MAX);
    }

    // GPU allocation and data trasnfer 
    float *d_input, *d_output, *d_weights;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_weights, total_weights * sizeof(float));
    
    cudaMemcpy(d_input, layers[0], N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, total_weights * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    for (int t = 0; t < K - 1; t++) {
        int current_layer_size = N - t * (R - 1);
        int next_layer_size = N - (t + 1) * (R - 1);

        // Numero di thread per blocco
        numBlocks = (next_layer_size + threadsPerBlock - 1) / threadsPerBlock;
        
        // Chiamata al kernel CUDA
        compute_layer<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_weights, bias, next_layer_size, R, offset);
        cudaDeviceSynchronize(); // Wait for the GPU to finish
    
        // Aggiorna l'offset
        offset += current_layer_size * R;
    }
    float* gpuOutput = malloc(layer_size * sizeof(float));
    // Sincronizza e copia i risultati
    cudaMemcpy(gpuOutput, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output Layer (first and last 10\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", gpuOutput[K - 1][i]);
    }
    printf("\n ...\n");
    for (int i = N - (K - 1) * (R - 1) - 10; i < N - (K - 1) * (R - 1); i++) {
        printf("%.4f ", gpuOutput[K - 1][i]);
    }   

    // Deallocazione memoria
    for (int t = 0; t < K; t++) {
        free(layers[t]);
    }
    free(layers);
    free(weights);
    
    // Deallocazione memoria sulla GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
}