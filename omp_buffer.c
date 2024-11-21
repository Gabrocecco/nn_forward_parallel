/****************************************************************************
 * Compile with:
 * gcc -fopenmp omp_buffer.c -o omp_buffer -lm
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp_buffer 7000000 100
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "hpc.h"

/* Global constants */
int R = 3; /* Kernel lenght, costant for all layers */

float sigmoid(float x) {
    // return 1.0 / (1.0 + exp(-x));
    return x / (1 + abs(x));
}

/* Fills n x n square matrix m with random values */
void fill( float* m, int n )
{
    int i, j;
    for (i=0; i<n; i++) {
        m[i] = (float)rand() / RAND_MAX;
    }
}

void fill_zeros( float* m, int n ){
    int i, j;
// #pragma omp parallel for
    for (i=0; i<n; i++) {
        m[i] = (float)0;
    }
}

void forward_propagation(int K, int R, int* layers_neuron_number, float* V, float* W, float* B) {
    int new_value_index = layers_neuron_number[0];  /* Start from the first neuron of layer 1 */
    int old_value_index = 0;  /* Start from the first neuron of the input layer (layer 0) */
    int weight_index = 0;     /* All weights are stored contiguously */

    for (int k = 1; k < K; k++) { // Non <= K

#pragma omp parallel for schedule(static)
        for (int i = 0; i < layers_neuron_number[k]; i++) { /* For every output neutron y_i*/
            float sum = 0.0;
            for (int r = 0; r < R; r++) {   /* Iterate on R last layer neutrons for calculate the sum  */
                int weight_idx = weight_index + (i * R) + r;    /* Local layer index of weights, for process*/ 
                int input_index = old_value_index + i + r;
                sum += V[input_index] * W[weight_index];
            }
            sum += B[k - 1];
            V[new_value_index + i] = sigmoid(sum);
        }
        
        weight_index += layers_neuron_number[k] * R;
        old_value_index = new_value_index;
        new_value_index += layers_neuron_number[k];
    }
}

/* Computer the Y_k layer outputs */
void step_forward(float* input_buffer, float* output_buffer, int k, int N, float* W, float* B, int weight_index) {
    int output_layer_size = N - k * (R - 1);  /* Number of neurons in current layer */
    
    int input_buffer_index = 0;  
    int output_buffer_index = 0; 
    /* For every neuron in output layer*/
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (N - k*(R - 1)); i++) { /* For every output neutron y_i in the output buffer)*/
        float sum = 0;
        int weight_idx = weight_index + i * R;  // Offset for the weight vector

        for (int r = 0; r < R; r++) {
            sum += input_buffer[i + r] * W[weight_idx + r];
        }
        sum += B[k - 1];  
        output_buffer[i] = sigmoid(sum);  
    }
}


int main( int argc, char *argv[] )
{
    float tstart, tstop;
    /* Initialized random generator */
    srand(time(NULL));
    /* Constants passed as argoument*/
    int N;  /* N = Number of neurons on layer 0 */
    int K;  /* K = Number of layers in the network */
    int N_neurons; /* Numbers of neurons from leyer 1 to layer K-1 */
    float *input_buffer, *output_buffer, *W, *B;  /* Data */
    int *layers_neuron_number;  /* Array with the numbers of neuron for each layer*/
    
    if (argc == 3) {
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }
    else return -1;
    
    printf("N = %d K = %d R = %d\n", N, K, R);

    omp_set_num_threads(atoi(getenv("OMP_NUM_THREADS"))); // Usa il numero di thread specificato
    printf("P = %s\n", getenv("OMP_NUM_THREADS"));
    /* Compute the total amount of neurons from input layer
        to K-1 layer.
    */
    layers_neuron_number = (int*)malloc( K * sizeof(int) );  
    N_neurons = 0;
    for(int t = 0; t < K; t++){
        N_neurons += (N - t*(R - 1));
        layers_neuron_number[t] = (N - t*(R - 1));
    }
    printf("Total neurons: %d \n", N_neurons);    

    /* Data generation with random float values, we need:
        - Layer 0: N input values
        - For each neuron we need 3 weights
        - In total we have N_neurons * 3 weights
        - For each layer we have 1 bias value 
         */
    // I = (float*)malloc( N * sizeof(float) );  /* Input layer vector */
    W = (float*)malloc( N_neurons * R * sizeof(float) ); /* Weights vector */
    B = (float*)malloc( (K - 1) * sizeof(float) ); /* Bias vector */
    input_buffer = (float*)malloc( N * sizeof(float) ); 
    output_buffer = (float*)malloc( (N - (R-1)) * sizeof(float) ); 
    if (input_buffer == NULL || output_buffer == NULL || W == NULL || B == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    tstart = hpc_gettime();
    fill(input_buffer, N);
    fill(output_buffer, (N - (R-1)));
    fill(W, N_neurons * 3);
    fill(B, K - 1);
    tstop = hpc_gettime();
    printf("Data preparation execution time %f\n", tstop - tstart); 


    // printf("\nINPUT_VECTOR (I), size %d\n ", N);
    // printf("\nWEIGHTS_VECTOR (W), size %d\n", N_neurons * 3);
    // printf("\nBIAS_VECTOR (B), size %d\n", K-1);
    // printf("\n\n");

    float oneCoreReference = 0.0;
    float speedup = 1;
    float lastspeedup = 1;
    float difference = 0;
    float best = 1;
    int bestP = 1;
    float bestDiff = 0;
    for(int p=1; p<=14; p++){
        omp_set_num_threads(p);
        printf("P=%d\n", p);


        tstart = hpc_gettime();
        int weight_index = 0;  // Start from the start of weights vector
        for (int k = 1; k < K; k++) {   // for each output vector
            step_forward(input_buffer, output_buffer, k, N, W, B, weight_index);
            // Switch buffer 
            float* temp = input_buffer;
            input_buffer = output_buffer;
            output_buffer = temp;   // we use the old input buffer as the new ouptu buffer
            // Update of the weight array index 
            weight_index += (N - k * (R - 1)) * R;
        }
        tstop = hpc_gettime();



        if(p==1){
            oneCoreReference = tstop - tstart;
            printf("Time: %f REFERENCE %f\n", tstop - tstart, speedup); 
        } else{
            lastspeedup = speedup;
            speedup = (oneCoreReference / (tstop - tstart)) * 100;
            difference = speedup - best;
            if(speedup > best){
                best = speedup;
                bestP = p;
                bestDiff = difference;
            }
            printf("Time: %f +%.2f%% (diff from best: +%.2f%%) \n", tstop - tstart, speedup, difference); 
        }
    }
    printf("\nBest speedup: +%.2f%% (diff from best: +%.2f%%), P=%d\n", best, bestDiff,bestP);

    free(input_buffer);
    free(output_buffer);
    free(layers_neuron_number);
    free(W);
    free(B);

    return 0;
}
