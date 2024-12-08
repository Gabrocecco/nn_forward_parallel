#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

const float bias = 0.1; // Constant bias 

// Sigmoid function, simple version 
__device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

// Kernel per il calcolo della prossima layer output
__global__ void compute_layer(float *input, float *output, float *weights, float bias, int N, int R, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
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

    // ./cuda <n_input_neurons> <R> <n_layers>
    if (argc != 4) {
        printf("Usage: %s <N> <R> <K>\n", argv[0]);
        return -1;
    }

    // Read input params 
    int N = atoi(argv[1]);
    int R = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("N=%d, R=%d, K=%d\n", N, R, K);

    // Calcola il numero totale di pesi
    int total_weights = 0;
    int layer_size;
    for (int t = 0; t < K - 1; t++) {   // per K-1 livelli (senza pesi per il livello di input)
        layer_size = N - t * (R - 1);   // numero di pesi nel livello corrente
        total_weights += layer_size * R;    // abbiamo R pesi unici per ogni neurone 
    }
    printf("Last layer size: %d\n", layer_size);
    printf("Numero totale di pesi: %d\n", total_weights);

    //--------------------------------------------------------PREPARAZIONE DEI DATI--------------------------------------------------------------
    printf("\n-----------------PREPARAZIONE DEI DATI-----------------\n");
    tstart = (float)clock() / CLOCKS_PER_SEC;
    // Allocazione dei dati
    float **layers = (float **)malloc(K * sizeof(float *)); // usiamo K array per i valori dei layer
    float *weights = (float *)malloc(total_weights * sizeof(float));    // un array unico per i pesi

    tot_number_of_bytes_allocated += (K * sizeof(float *)) + (total_weights * sizeof(float));

    // Allocazione dei valori dei layer e inizializzazione del layer di input
    for (int t = 0; t < K; t++) {
        int layer_size = N - t * (R - 1);
        layers[t] = (float *)malloc(layer_size * sizeof(float));    // allocazione per i valori di ciascun livello
        tot_number_of_bytes_allocated += (layer_size * sizeof(float));
        if (t == 0) {  // inizializzazione solo per il primo layer (input)
            for (int i = 0; i < layer_size; i++) {
                layers[0][i] = ((float)rand() / RAND_MAX);
            }
        }
    }

    // Inizializzazione dei pesi
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

    // Numero di thread per blocco
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

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

    // Lancio del kernel
    int offset = 0;
    for (int t = 0; t < K - 1; t++) {
        int current_layer_size = N - t * (R - 1);
        int next_layer_size = N - (t + 1) * (R - 1);

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
