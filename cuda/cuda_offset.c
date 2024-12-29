/****************************************************************************
 * Compile with:
 * gcc -fopenmp omp.c -o omp -lm -std=c99 -Wall -Wpedantic
 *
 * Run with:
 * ./omp N R K 
 *
 ****************************************************************************/
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <omp.h>
 #include "hpc.h"
 #include <sys/resource.h>

 const float bias = 0.1; // Constant bias 


// Sgimoid, simple version 
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

int main(int argc, char *argv[]) {
    float tstart, tstop;

    if (argc != 4) {
        printf("Usage: %s <N> <R> <K>\n", argv[0]);
        return -1;
    }

    /* Reading input parameters */
    int N = atoi(argv[1]); 
    int R = atoi(argv[2]); 
    int K = atoi(argv[3]); 
    printf("N=%d, R=%d, K=%d\n", N, R, K);

    /* We have K layers 
    
        layer_0 (input layer)  (N neurons)
        ...
        layer_t                 (N - t(R - 1) neurons)
        ...
        layer_K-1 (output layer) (N - K(R - 1) neurons)
    */
    
    // Computation of total number of weights 
    int total_weights = 0;
    int layer_size;
    int total_neurons = N;   // input layer has N neurons 
    for (int t = 1; t < K; t++) {   // from layer_1 to layer_K-1 (no weights for input layer)
        layer_size = N - t * (R - 1);   // number of neurons in the current layer
        total_neurons += layer_size; // update the number of total neurons
        total_weights += layer_size * R;    // we have R unique weights for each neuron 
    }
    int last_layer_size = layer_size;
    printf("Total neurons: %d\n", total_neurons);
    printf("Last layer size: %d\n", layer_size);
    printf("Number of weights: %d\n", total_weights);

    // we want to allocate two large sequential arrays, one for neurons activation 
    // and one for weights 
    int size = sizeof(float);   //both weights and activation are float's

    printf("CPU allocation...\n");
    tstart = hpc_gettime();
    float *activationsCPU = (float *)malloc(total_neurons * size);
    float *weightsCPU = (float *)malloc(total_weights * size);
    tstop = hpc_gettime();
    printf("Data allocation time: %f\n", tstop - tstart);
    printf("Allocating %d bytes (%d MB) of activations.\n Allocating %d bytes (%d MB) of weights.\n Tot: %d bytes (%d MB)\n",(total_neurons * size), (total_neurons * size / 1000000), (total_weights * size), (total_weights * size / 1000000), ((total_weights + total_neurons) * size), ((total_weights + total_neurons) * size / 1000000));
    // we need to initialize at random values the N actications of input layer
    // and all weights values 

    printf("CPU values initialization...\n");
    tstart = hpc_gettime();
    // Input layer initialization 
    for (int i = 0; i < N; i++) {
        activationsCPU[i] = ((float)rand() / RAND_MAX);
    }
    // printf("Input values:  \n");
    // for (int i = 0; i < N; i++) {
    // printf("%.4f ", activationsCPU[i]);
    // }
    // Weigths initialization 
    for (int i = 0; i < total_weights; i++) {
        weightsCPU[i] = ((float)rand() / RAND_MAX);
    }
    // printf("\nAll weights:  \n");
    // for (int i = 0; i < total_weights; i++) {
    //     printf("%.4f ", weightsCPU[i]);
    // }
    // printf("\nLast 5 weights:  \n");
    // for (int i = total_weights-5; i < total_weights; i++) {
    //     printf("%.4f ", weightsCPU[i]);
    // }
    tstop = hpc_gettime();
    printf("\nData initialization time: %f\n", tstop - tstart);


    int activations_offset = 0;
    int weights_offset = 0;
    tstart = hpc_gettime();
    // serial time output 
    for (int t = 1; t < K; t++) {   // from layer 1 to layer K-1
        int current_layer_size = N - (t-1) * (R - 1);
        int next_layer_size = N - t * (R - 1);
        // printf("t=%d\ncurrent_layer_size=%d\nnext_layer_size=%d\n\n", t, current_layer_size, next_layer_size);
        // we uodate the index of the first ouput neuron in next layer
        int output_idx = activations_offset + current_layer_size;    

        float sum;
        // for every output
        for(int i = 0; i < next_layer_size; i++){
            sum = 0.0;
            // for R weights and activation values 
            for(int r = 0; r < R; r++){
                sum += activationsCPU[activations_offset + i + r] * weightsCPU[weights_offset + r];
            }
            activationsCPU[output_idx + i] = sigmoid(sum);
            // activationsCPU[output_idx + i] = sum;
            // printf("y_%d = %.4f \n",i, activationsCPU[output_idx + i]);
            // update the weights offset, we never use a single weights more then once
            weights_offset += R;
        }
        // update the activation offset at the first neuron of the next input layer
        activations_offset += current_layer_size;
    }
    tstop = hpc_gettime();
    printf("Computer time CPU: %f\n", tstop - tstart);

    // we can print the last layer 
    // printf("Last 5 activations:  \n");
    // for (int i = total_neurons - 5; i < total_neurons; i++) {
    //     printf("%.4f ", activationsCPU[i]);
    // }

    printf("Last layer size: %d\n", last_layer_size);
    // printf("Output layer:  \n");
    // for (int i = total_neurons- last_layer_size; i < total_neurons; i++) {
    //     printf("%.4f ", activationsCPU[i]);
    // }
    // printf("\n");

    printf("Output layer (last 10):  \n");
    for (int i = total_neurons-10; i < total_neurons; i++) {
        printf("%.4f ", activationsCPU[i]);
    }
    printf("\n");

    free(activationsCPU);
    free(weightsCPU);

    

    
}

    