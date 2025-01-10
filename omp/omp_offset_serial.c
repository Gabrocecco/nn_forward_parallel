/****************************************************************************
 * Compile with:
 * gcc -fopenmp omp_offset_serial.c -o omp_offset_serial -lm -std=c99 -Wall -Wpedantic
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp_offset_serial 10000000 3 10 
 *
 ****************************************************************************/
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <omp.h>
 #include "hpc.h"
 #include <sys/resource.h>

const float bias = 0.1; // Constant bias for all layers

// Sgimoid, simple version 
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void compute_layer(float *activations,  // activations array 
                   float *weights,  // weights array
                   unsigned long int next_layer_size, // output layer size
                   int R,   // number of weights for each output neuron
                   unsigned long int activations_offset,  // offset of the first neuron of the input layer
                   unsigned long int weights_offset,  // offset for the weights
                   unsigned long int output_idx   // index of the first output neuron
                )
{
    float sum;
    // for every output
    #pragma omp parallel for schedule(static) private(sum)
    for(unsigned long int i = 0; i < next_layer_size; i++){
        sum = 0.0;
        // for R weights and activation values 
        for(int r = 0; r < R; r++){
            sum += activations[activations_offset + i + r] * weights[weights_offset + (i * R) + r];
            // printf("    x_i = %.4f * w = %.4f\n", activations[activations_offset + i + r], weights[weights_offset + (i * R) + r]);
        }
        activations[output_idx + i] = sigmoid(sum + bias);
        // printf("y_%d = %.4f \n",i, activations[output_idx + i]);
    }
}

int main(int argc, char *argv[]) {
    float tstart, tstop;

    if (argc != 4) {
        printf("Usage: %s <N> <R> <K>\n", argv[0]);
        return -1;
    }

    /* Reading input parameters */
    unsigned long int N = atoi(argv[1]); 
    int R = atoi(argv[2]); 
    int K = atoi(argv[3]); 
    // printf("N=%d, R=%d, K=%d\n", N, R, K);

    /* We have K layers 
    
        layer_0 (input layer)  (N neurons)
        ...
        layer_t                 (N - t(R - 1) neurons)
        ...
        layer_K-1 (output layer) (N - K(R - 1) neurons)
    */
    
    // Computation of total number of weights 
    unsigned long int total_weights = 0;
    unsigned long int layer_size;
    unsigned long int total_neurons = N;   // input layer has N neurons 
    for (int t = 1; t < K; t++) {   // from layer_1 to layer_K-1 (no weights for input layer)
        layer_size = N - t * (R - 1);   // number of neurons in the current layer
        total_neurons += layer_size; // update the number of total neurons
        total_weights += layer_size * R;    // we have R unique weights for each neuron 
    }
    unsigned long int last_layer_size = layer_size;
    // printf("Total neurons: %d\n", total_neurons);
    // printf("Last layer size: %d\n", layer_size);
    printf("Number of weights: %lu\n", total_weights);

    // we want to allocate two large sequential arrays, one for neurons activation 
    // and one for weights 
    int size = sizeof(float);   //both weights and activation are float's

    // printf("CPU allocation...\n");
    tstart = hpc_gettime();
    float *activations = (float *)malloc(total_neurons * size);
    float *weights = (float *)malloc(total_weights * size);
    tstop = hpc_gettime();
    // printf("Data allocation time: %f\n", tstop - tstart);
    // printf("Allocating %d bytes (%d MB) of activations.\n Allocating %d bytes (%d MB) of weights.\n Tot: %d bytes (%d MB)\n",(total_neurons * size), (total_neurons * size / 1000000), (total_weights * size), (total_weights * size / 1000000), ((total_weights + total_neurons) * size), ((total_weights + total_neurons) * size / 1000000));
    // we need to initialize at random values the N actications of input layer
    // and all weights values 

    // printf("CPU values initialization...\n");
    tstart = hpc_gettime();
    // Input layer initialization 
    for (unsigned long int i = 0; i < N; i++) {
        activations[i] = ((float)rand() / RAND_MAX);
    }
    // printf("Input values:  \n");
    // for (int i = 0; i < N; i++) {
    // printf("%.4f ", activations[i]);
    // }
    // Weigths initialization
    for (unsigned long int i = 0; i < total_weights; i++) {
        weights[i] = ((float)rand() / RAND_MAX);
    }
    // printf("\nAll weights:  \n");
    // for (int i = 0; i < total_weights; i++) {
    //     printf("%.4f ", weights[i]);
    // }
    // printf("\nLast 5 weights:  \n");
    // for (int i = total_weights-5; i < total_weights; i++) {
    //     printf("%.4f ", weights[i]);
    // }
    tstop = hpc_gettime();
    // printf("\nData initialization time: %f\n", tstop - tstart);

    unsigned long int activations_offset = 0;
    unsigned long int weights_offset = 0;
    tstart = hpc_gettime();
    // serial time output 
    int max_number_of_threads = omp_get_max_threads();
    // printf("MAX NUMBER OF THREADS: %d\n", max_number_of_threads);
    for (int t = 1; t < K; t++) {   // from layer 1 to layer K-1
        unsigned long int current_layer_size = N - (t-1) * (R - 1);
        unsigned long int next_layer_size = N - t * (R - 1);
        // printf("t=%d\ncurrent_layer_size=%d\nnext_layer_size=%d\n\n", t, current_layer_size, next_layer_size);
        // we uodate the index of the first ouput neuron in next layer
        unsigned long int output_idx = activations_offset + current_layer_size;    

        compute_layer(activations, weights, next_layer_size, R, activations_offset, weights_offset, output_idx);

        // update the activation offset at the first neuron of the next input layer
        activations_offset += current_layer_size;
        weights_offset += next_layer_size * R;
    }
    tstop = hpc_gettime();
    printf("Compute time CPU: %.5f\n", (float)tstop - tstart);

    // we can print the last layer 
    // printf("Last 5 activations:  \n");
    // for (int i = total_neurons - 5; i < total_neurons; i++) {
    //     printf("%.4f ", activations[i]);
    // }

    // printf("Last layer size: %d\n", last_layer_size);
    // printf("Output layer:  \n");
    // for (int i = total_neurons- last_layer_size; i < total_neurons; i++) {
    //     printf("%.4f ", activations[i]);
    // }
    // printf("\n");

    // printf("Output layer (last 10):  \n");
    // for (int i = total_neurons-10; i < total_neurons; i++) {
    //     printf("%.4f ", activations[i]);
    // }
    // printf("\n");

    free(activations);
    free(weights);
}

    