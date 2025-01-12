import matplotlib.pyplot as plt
import pandas as pd

# Leggi i risultati dal file di testo
data = pd.read_csv('results.txt')

# Crea il grafico
plt.figure(figsize=(10, 6))
plt.plot(data['Blocksize'], data['Time(ms)'], marker='o')
plt.xlabel('Blocksize')
plt.ylabel('Time (ms)')
plt.title('Compute Time vs Blocksize')
plt.grid(True)
plt.savefig('results_plot.png')
plt.show()