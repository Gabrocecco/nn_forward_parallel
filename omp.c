/****************************************************************************
 * Compile with:
 * gcc -fopenmp omp.c -o omp -lm 
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

// Utility function for extarct peak memory usage
void print_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("Peak memory usage: %.2f MB\n", usage.ru_maxrss / 1024.0);
}

// Forward funcion for calclulate next layer output
void compute_layer( float *input,   // pointer to the array of the input layer 
                    float *output,  // pointer to the array of the output layer, for storing the results 
                    float *weights, // pointe to the global array of weights 
                    float bias, // bias, constant for all computations 
                    int N,  // number of neurons in output layer
                    int R,  // constant R 
                    int offset  // offset for accessing weights array for the current layer 
                ) 
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {   // for every output neuron
        float sum = 0.0;
        for (int r = 0; r < R; r++) {   // R computations with R differente weights for each output neuron 
            sum += input[i + r] * weights[offset + i * R + r];
        }
        output[i] = sigmoid(sum + bias);
    }
}

int main(int argc, char *argv[]) {
    float tstart, tstop;
    long tot_number_of_bytes_allocated = 0;

    if (argc != 4) {
        printf("Usage: %s <N> <R> <K>\n", argv[0]);
        return -1;
    }

    /* Reading input parameters */
    int N = atoi(argv[1]); 
    int R = atoi(argv[2]); 
    int K = atoi(argv[3]); 
    printf("N=%d, R=%d, K=%d\n", N, R, K);
    
    // Computation of total number of weights 
    int total_weights = 0;
    for (int t = 0; t < K - 1; t++) {   // for K-1 layers (no weights for input layer)
        int layer_size = N - t * (R - 1);   // number of weights in the current layer
        total_weights += layer_size * R;    // we have R unique weights for each neuron 
    }
    printf("Nummber of weights: %d\n", total_weights);
    
    //--------------------------------------------------------DATA PREPARATION --------------------------------------------------------------
    printf("\n-----------------DATA PREPARATION-----------------\n");
    tstart = hpc_gettime();
    // Data allocation 
    float **layers = (float **)malloc(K * sizeof(float *)); // we use K arrays for the layer value, one for each layer
    float *weights = (float *)malloc(total_weights * sizeof(float));    // we use a single large array for storing weights sequentially
    printf("Allocating %ld bytes (%ld MB) of weights.\n", (total_weights * sizeof(float)), (total_weights * sizeof(float) / 1000000));

    tot_number_of_bytes_allocated += (K * sizeof(float *)) + (total_weights * sizeof(float));

    /* Allocation of layers values and initialization of input layer */
    long number_of_values = 0;
    for (int t = 0; t < K; t++) {
        int layer_size = N - t * (R - 1);
        layers[t] = (float *)malloc(layer_size * sizeof(float));    // data allocation of values for each of K layers
        tot_number_of_bytes_allocated += (layer_size * sizeof(float));
        // We initailize only the input layer
        if (t == 0) {
            for (int i = 0; i < layer_size; i++) {
                layers[0][i] = ((float)rand() / RAND_MAX);
            }
        }
    }

    // Weigths initialization 
    for (int i = 0; i < total_weights; i++) {
        weights[i] = ((float)rand() / RAND_MAX);
    }
    tstop = hpc_gettime();
    printf("Data preparation time: %f\n", tstop - tstart);
    printf("Total number of bytes allocated: %ld (%ld MB)\n",tot_number_of_bytes_allocated, tot_number_of_bytes_allocated/1000000);


    /* ----------------COMPUTATION---------------------------------------------------------------------------------------------*/
    printf("\n-----------------COMPUTUTATION-----------------\n");
    float serial_time;
    float best_time;
    int best_p = 1;
    float best_speedup;
    float current_time;
    for(int p=1; p <= 20; p=p*2){
        omp_set_num_threads(p);
        tstart = hpc_gettime();   
        int offset = 0; // Offset per accedere ai pesi del livello corrente
        for (int t = 0; t < K - 1; t++) {   // for K-1 layers 
            int current_layer_size = N - t * (R - 1);
            int next_layer_size = N - (t + 1) * (R - 1);

            compute_layer(layers[t], layers[t + 1], weights, bias, next_layer_size, R, offset);

            // Aggiorna l'offset per i pesi
            offset += current_layer_size * R;
        }
        tstop = hpc_gettime();   
        // printf("Inference time: %f\n", tstop - tstart);
        current_time = tstop - tstart;
        if(p==1){
            serial_time = current_time;
            best_time = current_time;
            best_speedup = 1;
        }
        if(best_time > current_time){
            best_time = tstop - tstart;
            best_p = p;
            best_speedup = (serial_time)/(tstop - tstart);
        }
        printf("P=%d, time: %.4f, speedup: %.3f, \n", p, tstop - tstart, (serial_time)/(tstop - tstart));

        // Printing first and last 10 values of output layer 
        // printf("Output Layer (first and last 10\n");
        // for (int i = 0; i < 10; i++) {
        //     printf("%.4f ", layers[K - 1][i]);
        // }
        // printf("\n ...\n");
        // for (int i = N - (K - 1) * (R - 1) - 10; i < N - (K - 1) * (R - 1); i++) {
        //     printf("%.4f ", layers[K - 1][i]);
        // }
        printf("\n");
    }
    printf("P=%d best time %.4f: best speedup: %.3f\n", best_p, best_time, best_speedup);

    // Printf peak memory usage
    print_memory_usage();

    // Memory deallocation
    for (int t = 0; t < K; t++) {
        free(layers[t]);
    }
    free(layers);
    free(weights);

    return 0;
}

