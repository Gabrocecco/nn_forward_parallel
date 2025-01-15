import matplotlib.pyplot as plt
import numpy as np

# Dati 2080
N = np.array([31250, 62500, 125000, 250000, 500000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000])
data_prep_time = np.array([121.110687, 194.262817, 366.209381, 757.720886, 1558.889648, 3243.483154, 6720.373047, 10422.586914, 14221.101562, 18059.853516, 22180.697266, 26381.396484])
compute_time = np.array([0.978880, 1.176288, 1.363968, 1.880064, 2.927872, 4.820640, 8.754336, 12.715968, 16.588768, 20.569984, 24.742720, 28.652224])
#dati l40
N = np.array([31250, 62500, 125000, 250000, 500000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000])
data_prep_time = np.array([121.110687, 194.262817, 366.209381, 757.720886, 1558.889648, 3243.483154, 
                           6720.373047, 10422.586914, 14221.101562, 18059.853516, 22180.697266, 26381.396484])
compute_time = np.array([0.978880, 1.176288, 1.363968, 1.880064, 2.927872, 4.820640, 8.754336, 12.715968, 
                         16.588768, 20.569984, 24.742720, 28.652224])


# Convertiamo i tempi da millisecondi a secondi per una scala più leggibile
data_prep_time /= 1000
compute_time /= 1000

# Creazione dei grafici
plt.figure(figsize=(12, 6))

# Grafico del tempo di preparazione dei dati
plt.subplot(1, 2, 1)
plt.plot(N, data_prep_time, marker='o', label='Data Preparation Time', color='blue')
plt.title('Tempo di preparazione dei dati')
plt.xlabel('Numero di neuroni di input (N)')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.legend()

# Grafico del tempo di calcolo
plt.subplot(1, 2, 2)
plt.plot(N, compute_time, marker='o', label='Compute Time', color='green')
plt.title('Tempo di calcolo')
plt.xlabel('Numero di neuroni di input (N)')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.legend()

# Mostra i grafici
plt.tight_layout()
plt.show()
