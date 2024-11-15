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

/* Global constants */
int R = 3; /* Kernel lenght, costant for all layers */


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
    for (i=0; i<n; i++) {
        m[i] = (double)0;
    }
}
// void fill( double* m, int n )
// {
//     int i, j;
//     for (i=0; i<n; i++) {
//         for (j=0; j<n; j++) {
//             m[i*n + j] = (double)rand() / RAND_MAX;
//         }
//     }
// }

int main( int argc, char *argv[] )
{
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

    printf("N = %d\n K = %d\n R = %d\n", N, K, R);

    /* Compute the total amount of neurons from layer 1 to layer k-1,
       knowing that layer t has (N - t(R-1)) neurons.
    */
    layers_neuron_number = (int*)malloc( K * sizeof(int) );  
    layers_neuron_number[0] = N;
    N_neurons = 0;
    for(int t = 1; t < K; t++){
        int n_neurons_current_layer = (N - t*(R - 1));
        if (n_neurons_current_layer <= 0) {
            printf("Error: Invalid neuron count in layer %d.\n", t);
            return -1;
        }
        N_neurons += (N - t*(R - 1));
        layers_neuron_number[t] = (N - t*(R - 1));

        printf("Neurons in layer %d: %d\n", t, (N - t*(R - 1)));
    }
    printf("Total neurons from layer 1 to layer K: %d \n", N_neurons);

    /* Data generation with random double values, we need:
        - Layer 0: N input values
        - For each neuron we need 3 weights
        - In total we have N_neurons * 3 weights
        - For each layer we have 1 bias value 
         */
    // I = (double*)malloc( N * sizeof(double) );  /* Input layer vector */
    W = (double*)malloc( N_neurons * 3 * sizeof(double) ); /* Weights vector */
    B = (double*)malloc( (K - 1) * sizeof(double) ); /* Bias vector */
    V = (double*)malloc( N_neurons * sizeof(double) ); /* Neurons value vector  */

    fill_zeros(V, N_neurons); // hidden layers has value 0 at the start
    fill(V, N); // we allocate the first N neurons value for the input layer 
    fill(W, N_neurons * 3);
    fill(B, K - 1);

    printf("\n\nINPUT_VECTOR (I), size %d\n ", N);
    // for(int i=0; i<N; i++){
    //     printf("%f ", I[i]);
    // }
    printf("\n\nWEIGHTS_VECTOR (W), size %d\n", N_neurons * 3);
    int index_W = 0;
    // for(int i=1; i < K ; i++){
    //     printf("    Layer: %d, (size %d) \n", i, layers_neuron_number[i] * 3);
    //     for(int j=0; j<layers_neuron_number[i] * 3; j++){
    //         printf("%f ", W[index_W]);
    //         index_W++;
    //     }
    //     printf("\n");
    // }
    printf("\nBIAS_VECTOR (B), size %d\n", K-1);
    // for(int i=0; i<K-1 ; i++){
    //     printf("%f ", B[i]);
    // }
    printf("\n\n");


    /* Serial version */

    /* For each layer */
    int new_value_index = N; // we start from first neuron of layer 1
    int old_value_index = 0;
    int weight_index = 0;   //all weights are used only 1 time and stored contigously 
    for(int k=1; k<=K; k++){
        /* For each neuron in the layer*/
        for(int i=0; i<layers_neuron_number[k]; i++){
            double sum = 0.0;
            // calculation of y_i
            for(int r=0; r < R; r++){
                int input_index = old_value_index + i + r; // Adjusted input index
                if (input_index < new_value_index) { // Bounds check
                    sum += V[input_index] * W[weight_index];
                    weight_index++;
                }
                printf("Layer %d, neuron y_%d, r=%d, value=%f\n", k, i, r, sum);
            }
            sum += B[k - 1]; // Add bias
            V[new_value_index + i] = sum; // Update current neuron
            printf("Layer %d, neuron y_%d = %f\n", k, i, sum);
        }
        old_value_index = new_value_index;
        new_value_index += layers_neuron_number[k]; // Move to next layer's start
    }

    free(layers_neuron_number);
    free(V);
    free(W);
    free(B);
    return 0;
}
