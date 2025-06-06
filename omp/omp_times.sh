#!/bin/bash

# Compila il programma OMP
gcc -fopenmp omp_offset_serial.c -o omp_offset_serial -lm -std=c99 -Wall -Wpedantic

# Array di valori di N da testare
Ns=(31250 62500 125000 250000 500000 1000000 2000000 3000000 4000000 5000000 6000000 7000000)

# File di output per i risultati
output_file="omp_times.txt"
echo "N,Time(s)" > $output_file

# Itera sui valori di N e esegui il programma
for N in "${Ns[@]}"; do
    echo "Esecuzione con N = $N"
    OMP_NUM_THREADS=20 ./omp_offset_serial $N 3 100 >> $output_file
done
