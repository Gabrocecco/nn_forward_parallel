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

// Kernel that calculates a layer output, spreding work across threads 
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

    if (idx < output_size) {    // never exeeds the number of output's neurons 
        float sum = 0.0;
        for (int r = 0; r < R; r++) {
            sum += input[idx + r] * weights[offset + idx * R + r];
        }
        output[idx] = sigmoid(sum + bias);
    }
}

int main(int argc, char *argv[]) {
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

    //--------------------------------------------------------DATA PREPARATION--------------------------------------------------------------
    printf("\n-----------------DATA PREPARATION-----------------\n");
    tstart = (float)clock() / CLOCKS_PER_SEC;
    // Data allocation on CPU 
    float **layers = (float **)malloc(K * sizeof(float *)); // we use K array's for value activations of the K layers
    float *weights = (float *)malloc(total_weights * sizeof(float));    // we use a single long array for weights

    tot_number_of_bytes_allocated += (K * sizeof(float *)) + (total_weights * sizeof(float));

    // Allocation for each activation layer 
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

    // Random weights initialization 
    for (int i = 0; i < total_weights; i++) {
        weights[i] = ((float)rand() / RAND_MAX);
    }
    tstop = (float)clock() / CLOCKS_PER_SEC;
    printf("Tempo di preparazione dei dati: %.2f secondi\n", tstop - tstart);
    printf("Numero totale di byte allocati: %ld (%ld MB)\n", tot_number_of_bytes_allocated, tot_number_of_bytes_allocated / 1000000);

    /* -----------------CALCOLO---------------------------------------------------------------------------------------------*/
    printf("\n-----------------CALCOLO-----------------\n");
    float serial_time;
    float best_time;
    float current_time;

    // Allocazione memoria sulla GPU
    float *d_input, *d_output, *d_weights;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_weights, total_weights * sizeof(float));

    cudaMemcpy(d_input, layers[0], N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, total_weights * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsed_time;

    // Creazione degli eventi per il timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    int threadsPerBlock = 512;
    int numBlocks;
    // Lancio del kernel
    int offset = 0;
    for (int t = 0; t < K - 1; t++) {
        int current_layer_size = N - t * (R - 1);
        int next_layer_size = N - (t + 1) * (R - 1);
        
        // Numero di thread per blocco
        numBlocks = (next_layer_size + threadsPerBlock - 1) / threadsPerBlock;
        
        // Chiamata al kernel CUDA
        compute_layer<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_weights, bias, next_layer_size, R, offset);
    
        // Aggiorna l'offset
        offset += current_layer_size * R;
    }

    // Sincronizzazione e misura del tempo
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Tempo di esecuzione del kernel: %.2f ms\n", elapsed_time);

    // Sincronizza e copia i risultati
    cudaMemcpy(layers[K - 1], d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Distruzione degli eventi
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Deallocazione memoria sulla GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);

    current_time = elapsed_time / 1000.0f;  // conversione a secondi
    printf("Tempo di esecuzione del kernel: %.4f secondi\n", current_time);

    // Stampa del picco di utilizzo della memoria
    // (aggiungi una funzione per ottenere questo valore, se necessario)
    // Printing first and last 10 values of output layer 
    printf("Output Layer (first and last 10\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", layers[K - 1][i]);
    }
    printf("\n ...\n");
    for (int i = N - (K - 1) * (R - 1) - 10; i < N - (K - 1) * (R - 1); i++) {
        printf("%.4f ", layers[K - 1][i]);
    }   
    // Deallocazione memoria
    for (int t = 0; t < K; t++) {
        free(layers[t]);
    }
    free(layers);
    free(weights);

    return 0;
}
