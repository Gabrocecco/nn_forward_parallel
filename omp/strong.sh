#!/bin/bash

# Nome del programma
PROGRAM="./omp_offset_serial"

# Parametri fissi
PARAM1=10000000
PARAM2=3
PARAM3=10

# File di output
OUTPUT_FILE="strong2.txt"

# Pulizia del file di output se esiste già
> $OUTPUT_FILE
echo "$PROGRAM $PARAM1 $PARAM2 $PARAM3"  | tee -a $OUTPUT_FILE
# Loop sui thread da 1 a 20
for THREADS in {1..20}; do
    echo "Esecuzione con OMP_NUM_THREADS=$THREADS" | tee -a $OUTPUT_FILE
    OMP_NUM_THREADS=$THREADS $PROGRAM $PARAM1 $PARAM2 $PARAM3 | tee -a $OUTPUT_FILE
    echo "---------------------------------------------" >> $OUTPUT_FILE
done

echo "Tutti i risultati sono stati salvati in $OUTPUT_FILE"
