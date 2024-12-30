import matplotlib.pyplot as plt

# Data extracted from your output
threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
compute_times = [
    0.125, 0.15625, 0.15625, 0.15625, 0.15625, 0.15625, 0.21875, 0.25, 0.25, 0.25, 0.25, 0.28125, 
    0.28125, 0.28125, 0.28125, 0.28125, 0.28125, 0.3125, 0.3125, 0.3125
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(threads, compute_times, marker='o', linestyle='-', color='b', label='Compute Time')
    
# Impostare i valori dell'asse x come interi
plt.xticks(threads)

# Labeling the axes
plt.xlabel('Number of Threads (P)', fontsize=12)
plt.ylabel('Compute Time (seconds) T(P)', fontsize=12)

# Adding a title
plt.title('Weak Scaling of the Problem with P threads ', fontsize=14)

# Log scale for the x-axis to better visualize the scaling
plt.xscale('linear')
plt.yscale('linear')

# Display grid
plt.grid(True)

# Show the plot
plt.legend()

plt.savefig("weak_time_plot.png")
plt.show()