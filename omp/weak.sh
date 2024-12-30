#!/bin/bash

# Nome del programma
PROGRAM="./omp_offset_serial"

# Parametri fissi
PARAM1=10000000
PARAM2=3
PARAM3=10

# File di output
OUTPUT_FILE="weak.txt"

# Pulizia del file di output se esiste già
> $OUTPUT_FILE
echo "$PROGRAM $PARAM1 $PARAM2 $PARAM3"  | tee -a $OUTPUT_FILE

# Loop sui thread da 1 a 20
for THREADS in {1..20}; do
    # Calcolo N scalato in base al numero di thread
    N=1000000
    N_SCALED=$((N * THREADS))  # Moltiplichiamo N per il numero di thread

    # Aggiungi il messaggio con N scalato
    echo "Esecuzione con OMP_NUM_THREADS=$THREADS problema scalato di $THREADS" | tee -a $OUTPUT_FILE
    
    # Passa N_SCALED al programma come parametro aggiuntivo
    OMP_NUM_THREADS=$THREADS $PROGRAM $N_SCALED $PARAM2 $PARAM3 | tee -a $OUTPUT_FILE
    
    # Separatore tra le esecuzioni
    echo "---------------------------------------------" >> $OUTPUT_FILE
done

echo "Tutti i risultati sono stati salvati in $OUTPUT_FILE"
