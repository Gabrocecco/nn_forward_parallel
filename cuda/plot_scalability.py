import matplotlib.pyplot as plt
import pandas as pd

# Leggi i risultati dal file di testo
data = pd.read_csv('scalability_results.txt')

# Crea il grafico
plt.figure(figsize=(10, 6))
plt.plot(data['N'], data['Time(ms)'], marker='o')
plt.xlabel('N')
plt.ylabel('Time (ms)')
plt.title('Compute Time vs Problem Size (N)')
plt.grid(True)
plt.savefig('scalability_plot.png')
plt.show()