/****************************************************************************
 * Compile with:
 * gcc -fopenmp omp.c -o omp
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp N K 
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "hpc.h"

/* Global constants */
int R = 3; /* Kernel lenght, costant for all layers */

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/* Fills n x n square matrix m with random values */
void fill( double* m, int n )
{
    int i, j;
    for (i=0; i<n; i++) {
        m[i] = (double)rand() / RAND_MAX;
    }
}

void fill_zeros( double* m, int n ){
    int i, j;
// #pragma omp parallel for
    for (i=0; i<n; i++) {
        m[i] = (double)0;
    }
}

void forward_propagation(int K, int R, int* layers_neuron_number, double* V, double* W, double* B) {
    int new_value_index = layers_neuron_number[0];  /* Start from the first neuron of layer 1 */
    int old_value_index = 0;  /* Start from the first neuron of the input layer (layer 0) */
    int weight_index = 0;     /* All weights are stored contiguously */

    for (int k = 1; k < K; k++) { // Non <= K
        for (int i = 0; i < layers_neuron_number[k]; i++) { /* For every output neutron y_i*/
            double sum = 0.0;
            for (int r = 0; r < R; r++) {   /* Iterate on R last layer neutrons for calculate the sum  */
                int input_index = old_value_index + i + r;
                sum += V[input_index] * W[weight_index++];
            }
            sum += B[k - 1];
            V[new_value_index + i] = sigmoid(sum);
        }
        old_value_index = new_value_index;
        new_value_index += layers_neuron_number[k];
    }
}

int main( int argc, char *argv[] )
{
    double tstart, tstop;
    /* Initialized random generator */
    srand(time(NULL));
    /* Constants passed as argoument*/
    int N;  /* N = Number of neurons on layer 0 */
    int K;  /* K = Number of layers in the network */
    int N_neurons; /* Numbers of neurons from leyer 1 to layer K-1 */
    double *I, *W, *B, *V;  /* Data */
    int *layers_neuron_number;  /* Array with the numbers of neuron for each layer*/
    
    if (argc == 3) {
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }
    else return -1;

    printf("N = %d K = %d R = %d\n", N, K, R);

    /* Data generation with random double values, we need:
        - Layer 0: N input values
        - For each neuron we need 3 weights
        - In total we have N_neurons * 3 weights
        - For each layer we have 1 bias value 
         */
    // I = (double*)malloc( N * sizeof(double) );  /* Input layer vector */
    W = (double*)malloc( N_neurons * R * sizeof(double) ); /* Weights vector */
    B = (double*)malloc( (K - 1) * sizeof(double) ); /* Bias vector */
    V = (double*)malloc( (N + N_neurons) * sizeof(double) ); /* Neurons value vector  */

    if (V == NULL || W == NULL || B == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    fill_zeros(V, N_neurons); // hidden layers has value 0 at the start
    fill(V, N); // we allocate the first N neurons value for the input layer 
    tstart = hpc_gettime();
    fill(W, N_neurons * 3);
    tstop = hpc_gettime();
    printf("Execution time %f\n", tstop - tstart); 

    fill(B, K - 1);

    printf("\nINPUT_VECTOR (I), size %d\n ", N);
    printf("\nWEIGHTS_VECTOR (W), size %d\n", N_neurons * 3);
    printf("\nBIAS_VECTOR (B), size %d\n", K-1);
    printf("\n\n");

    /* Serial version */
    /* For each layer */
    int new_value_index = N;  // Start from first neuron of layer 1
    int old_value_index = 0;  // Start from first neuron of linput layer (layer 0)
    int weight_index = 0;   // All weights are stored contiguously

    tstart = hpc_gettime();
    forward_propagation(K, R, layers_neuron_number, V, W, B);
    tstop = hpc_gettime();
    printf("Execution time %f\n", tstop - tstart); 

    free(layers_neuron_number);
    free(V);
    free(W);
    free(B);
    return 0;
}
