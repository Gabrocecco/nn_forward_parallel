#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00
#SBATCH --output last.out
#SBATCH --partition=l40
## TODO: change accordingly
# export EXE="cuda"

echo "=== Start CUDA program ==="
# Compila il programma CUDA
nvcc -o cuda cuda.cu

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
    result=$(./cuda $N $R $K $blocksize | grep "Compute time" | awk '{print $3}')
    echo "$blocksize,$result" >> $output_file
    echo "Blocksize = $blocksize, Time = $result ms"
done

echo "=== End of Job ==="