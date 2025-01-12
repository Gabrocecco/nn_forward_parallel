#!/bin/bash

# Compila il programma CUDA
nvcc -o cuda_program cuda.cu

# Parametri di input
N=10000
R=10
K=5

# Array di valori di blocksize da testare
blocksizes=(128 256 512 1024)

# File di output per i risultati
output_file="results.txt"
echo "Blocksize,Time(ms)" > $output_file

# Itera sui valori di blocksize e esegui il programma
for blocksize in "${blocksizes[@]}"; do
    echo "Esecuzione con blocksize = $blocksize"
    result=$(./cuda_program $N $R $K $blocksize | grep "Compute time" | awk '{print $3}')
    echo "$blocksize,$result" >> $output_file
    echo "Blocksize = $blocksize, Time = $result ms"
done