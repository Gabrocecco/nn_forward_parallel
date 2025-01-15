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
R=10
K=5
blocksize=1024

# Array di fattori di scala per N
scaling_factors=(1 2 4 8 16)

# File di output per i risultati
output_file="scalability_results.txt"
echo "N,Time(ms)" > $output_file

# Itera sui fattori di scala e esegui il programma
for factor in "${scaling_factors[@]}"; do
    N=$((10000 * factor))
    echo "Esecuzione con N = $N"
    result=$(./cuda $N $R $K $blocksize | grep "Compute time" | awk '{print $3}')
    echo "$N,$result" >> $output_file
    echo "N = $N, Time = $result ms"
done

echo "=== End of Job ==="
