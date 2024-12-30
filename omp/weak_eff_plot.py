import re
import matplotlib.pyplot as plt

# Funzione per leggere i dati dal file e calcolare l'efficienza
def read_and_plot_efficiency(file_path):
    threads = []
    compute_times = []

    # Leggere il file riga per riga
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Troviamo tutte le esecuzioni basandoci sul pattern "Esecuzione con OMP_NUM_THREADS"
        executions = re.findall(r"Esecuzione con OMP_NUM_THREADS=(\d+).*?Compute time CPU:\s+(\d+\.\d+)", content, re.DOTALL)
        
        # Per ogni esecuzione, estraiamo il numero di thread e il tempo di calcolo
        for execution in executions:
            num_threads = int(execution[0])
            compute_time = float(execution[1])
            threads.append(num_threads)
            compute_times.append(compute_time)
    for c in compute_times:
        print(c)
    # Tempo di esecuzione con 1 thread
    time_single_thread = compute_times[0]

    # Calcolare l'efficienza per ciascun numero di thread
    efficiency = [(time_single_thread / (time)) for i, time in enumerate(compute_times)]

    # Creare il grafico dell'efficienza
    plt.figure(figsize=(10, 6))
    plt.plot(threads, efficiency, marker='o', linestyle='-', color='g', label='Efficiency')

    # Aggiungere una linea orizzontale rossa all'altezza y=1
    plt.axhline(y=1, color='r', linestyle='--', label='Ideal Efficiency')

    # Impostare i valori dell'asse x come interi
    plt.xticks(threads)

    # Etichette degli assi
    plt.xlabel('Number of threads (P)', fontsize=12)
    plt.ylabel('Efficiency = T(1) / T(P) ', fontsize=12)

    # Titolo
    plt.title('Efficiency drop in weak scaling', fontsize=14)

    # Aggiungere una griglia
    plt.grid(True)

    # Mostrare il grafico
    plt.legend()
    plt.savefig("weak_eff_plot.png")
    plt.show()

# Esegui lo script per il file di esempio
file_path = 'weak.txt'  # Cambia con il percorso del tuo file
read_and_plot_efficiency(file_path)
