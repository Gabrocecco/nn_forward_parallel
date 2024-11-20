/****************************************************************************
 *
 * omp.c 
 *
 * Compile with:
 * gcc -fopenmp omp.c -o omp -lm
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp N K 
 *
 * (N = input size)
 * (K = layers)
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

/* Fills an array with random n doubles */
void fill_random( double* m, int n )
{
    int i;
#pragma omp parallel for
    for (i=0; i<n; i++) {
        m[i] = (double)rand() / RAND_MAX;
    }
}

/* Fills an array with n 0s */
void fill_zeros( double* m, int n )
{
    int i;
#pragma omp parallel for
    for (i=0; i<n; i++) {
        m[i] = (double)rand() / RAND_MAX;
    }
}

void forward_step(*last_layer_buffer, current_layer_buffer, *W, int layer_number){

}

int main( int argc, char *argv[] )
{
    /* Constants passed as argoument*/
    int N;  /* N = Number of neurons on layer 0 */
    int K;  /* K = Number of layers in the network */
    int N_neurons; /* Numbers of neurons from leyer 1 to layer K-1 */
    double *I, *W, *B;  /* Data */
    // int *layers_neuron_number;  /* Array with the numbers of neuron for each layer*/

    double tstart, tstop;

    /* Initialized random generator */
    srand(time(NULL));   
    if (argc == 3) {
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }
    else return -1;

    printf("N = %d K = %d R = %d\n", N, K, R);

    /* Compute the total amount of neurons from input layer
        to K-1 layer.
    */
    // layers_neuron_number = (int*)malloc( K * sizeof(int) );  
    N_neurons = 0;
    for(int t = 0; t < K; t++){
        N_neurons += (N - t*(R - 1));
        // layers_neuron_number[t] = (N - t*(R - 1));
    }
    printf("Total neurons: %d \n", N_neurons);

    /* Allocate memory for weights, 
        from layer 1 to layer K-1 we need R doubles for neuron.*/
    W = (double*)malloc( (N_neurons - N) * R * sizeof(double) );
    /* We want an unique bias for each layer, from layer 1 to layer K-1*/
    B = (double*)malloc( (K - 1) * sizeof(double) );
    /* We want N doubles for the input layer. */
    I = (double*)malloc( N * sizeof(double) );
    
    tstart = hpc_gettime();
    /* We want to randomize weights, biaseses and input values.*/
    fill_random(W, (N_neurons - N) * R);
    fill_random(B, K-1);
    fill_random(I, N);
    if (I == NULL || W == NULL || B == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }
    tstop = hpc_gettime();
    printf("Execution time for filling values %f\n", tstop - tstart); 

    for(int i=1; i < K; i++){
        forward_step();
    }

    free(W);
    free(B);
    free(I);

    return EXIT_SUCCESS;
}
