import matplotlib.pyplot as plt
import numpy as np
import os

data = []
with open('results.csv', 'r') as f:
    for line in f:                              #the file results.csv also contains some useless text that we should disregard
        parts = line.strip().split(',')                 #counts numbers of elements of row
        if len(parts) >= 4 and parts[0].isdigit():      #checks if the row actually has the correct data
            data.append([float(x) for x in parts])

data = np.array(data)
grid_sizes = np.unique(data[:, 0])

plt.figure(figsize=(9, 6))

for n in grid_sizes:
    subset = data[data[:, 0] == n]
    plt.loglog(subset[:, 1], subset[:, 2], marker='o', linewidth=2, label=f'n = {int(n)}')

#Execution time
plt.xlabel('Number of MPI Ranks', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Execution Time vs MPI Ranks', fontsize=14, fontweight='bold')
plt.xticks([1, 2, 4]) 
plt.legend(title='Grid Size')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('scalability.png', dpi=300)
print("Salvato: scalability.png")


#L2 error plots
subset_1rank = data[data[:, 1] == 1]
n_vals = subset_1rank[:, 0]
h_vals = 1.0 / (n_vals - 1)
errors = subset_1rank[:, 3]

plt.figure(figsize=(9, 6))
plt.loglog(h_vals, errors, marker='s', color='darkred', markersize=8, linewidth=2, label='L2 Error')


plt.xlabel('Grid Spacing $h$', fontsize=12)
plt.ylabel('Error ($L_2$ Norm)', fontsize=12)
plt.title('Error Analysis vs Grid Spacing', fontsize=14, fontweight='bold')
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

plt.savefig('error_L2.png', dpi=300)
print("Salvato: error_L2.png")
