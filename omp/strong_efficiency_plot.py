import matplotlib.pyplot as plt
import re

# Nome del file dei risultati
output_file = "strong2.txt"

# Liste per salvare i dati
threads = []
compute_times = []

# Pattern per trovare il numero di thread e il tempo di calcolo
thread_pattern = re.compile(r"Esecuzione con OMP_NUM_THREADS=(\d+)")
compute_time_pattern = re.compile(r"Compute time CPU:\s*([\d.]+)")

# Lettura del file
with open(output_file, "r") as file:
    for line in file:
        # Cerca il numero di thread
        thread_match = thread_pattern.search(line)
        if thread_match:
            threads.append(int(thread_match.group(1)))
        
        # Cerca il tempo di calcolo
        compute_time_match = compute_time_pattern.search(line)
        if compute_time_match:
            compute_times.append(float(compute_time_match.group(1)))

# Verifica che ci siano dati
if not threads or not compute_times:
    print("Nessun dato trovato nel file. Verifica il formato del file dei risultati.")
    exit()

# Calcolo dello speedup
if threads[0] != 1:
    print("Errore: il primo dato non corrisponde al tempo con 1 thread.")
    exit()

single_thread_time = compute_times[0]  # Tempo con un solo thread
speedup = [single_thread_time / time for time in compute_times]

# Calcolo dell'efficienza
efficiency = [s / t for s, t in zip(speedup, threads)]

# Creazione del grafico dello speedup
plt.figure(figsize=(10, 6))
plt.plot(threads, speedup, marker='o', color='g', label='Speedup')
plt.xlabel("Number of threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of threads")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Salvataggio e visualizzazione del grafico
# plt.savefig("speedup_plot.png")
# plt.show()

# Creazione del grafico dell'efficienza
plt.figure(figsize=(10, 6))
plt.plot(threads, efficiency, marker='o', color='g', label='Efficiency')
plt.xlabel("Number of threads (P)")
plt.ylabel("Efficiency E(P) = S(P) / P")
plt.title("(Strong scaling) Efficiency vs Number of threads")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Aggiungere una linea orizzontale rossa all'altezza y=1
plt.axhline(y=1, color='r', linestyle='--', label='Ideal Efficiency')
    
# Impostare i valori dell'asse x come interi
plt.xticks(threads)

# Salvataggio e visualizzazione del grafico
plt.savefig("strong_efficiency_plot.png")
plt.show()
