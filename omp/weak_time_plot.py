import matplotlib.pyplot as plt
import re

# Nome del file dei risultati
input_file = "weak.txt"

# Liste per salvare i dati
threads = []
compute_times = []

# Pattern per trovare il numero di thread e il tempo di calcolo
thread_pattern = re.compile(r"OMP_NUM_THREADS=(\d+)")
compute_time_pattern = re.compile(r"Compute time CPU:\s*([\d.]+)")

# Lettura del file
with open(input_file, "r") as file:
    for line in file:
        # Cerca il numero di thread
        thread_match = thread_pattern.search(line)
        if thread_match:
            threads.append(int(thread_match.group(1)))
        
        # Cerca il tempo di calcolo
        compute_time_match = compute_time_pattern.search(line)
        if compute_time_match:
            compute_times.append(float(compute_time_match.group(1)))

# Debugging: verificare la lunghezza delle liste
print(f"Threads: {len(threads)} elementi - {threads}")
print(f"Compute Times: {len(compute_times)} elementi - {compute_times}")

# Controllo della lunghezza delle liste
min_length = min(len(threads), len(compute_times))

# Allineare threads e compute_times alla stessa lunghezza
threads = threads[:min_length]
compute_times = compute_times[:min_length]

print(f"Dati aggiornati: Threads = {threads}, Compute Times = {compute_times}")

# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(threads, compute_times, marker='o', linestyle='-', color='b', label='Compute Time')

# Impostare i valori dell'asse x come interi
plt.xticks(threads)

# Etichette degli assi
plt.xlabel('Number of Threads (P)', fontsize=12)
plt.ylabel('Compute Time (seconds) T(P)', fontsize=12)

# Titolo
plt.title('Weak Scaling of the Problem with P Threads', fontsize=14)

# Aggiungere una griglia
plt.grid(True)

# Aggiungere una legenda
plt.legend()

# Salvataggio e visualizzazione del grafico
plt.savefig("weak_time_plot.png")
plt.show()

