import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(csv_filename, solver_name):
    data = []
    
    if not os.path.exists(csv_filename):
        print(f"File {csv_filename} not found. Launch the script from test/data/")
        return

    with open(csv_filename, 'r') as f:      #the csv files have some "useless text" rows, so we ignore them
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4 and parts[0].isdigit():
                data.append([float(x) for x in parts])

    data = np.array(data)
        
    grid_sizes = np.unique(data[:, 0])

    #Execution times plots
    plt.figure(figsize=(9, 6))

    for n in grid_sizes:
        subset = data[data[:, 0] == n]
        plt.plot(subset[:, 1], subset[:, 2], marker='o', linewidth=2, label=f'n = {int(n)}')

    plt.xlabel('Number of MPI Ranks', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title(f'Execution Time vs MPI Ranks ({solver_name})', fontsize=14, fontweight='bold')
    plt.xticks([1, 2, 4], ['1', '2', '4']) 
    plt.legend(title='Grid Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    out_scalability = f'scalability_{solver_name}.png'
    plt.savefig(out_scalability, dpi=300)
    print(f"Salvato: {out_scalability}")


    #L2 error graphs
    subset_1rank = data[data[:, 1] == 1]
    n_vals = subset_1rank[:, 0]
    h_vals = 1.0 / (n_vals - 1)
    errors = subset_1rank[:, 3]

    plt.figure(figsize=(9, 6))
    plt.loglog(h_vals, errors, marker='s', color='darkred', markersize=8, linewidth=2, label='L2 Error')

    plt.xlabel('Grid Spacing $h$', fontsize=12)
    plt.ylabel('Error ($L_2$ Norm)', fontsize=12)
    plt.title(f'Error Analysis vs Grid Spacing ({solver_name})', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    out_error = f'error_L2_{solver_name}.png'
    plt.savefig(out_error, dpi=300)
    print(f"Salvato: {out_error}")
    
    plt.close('all')


plot_results('../results_jacobi.csv', 'Jacobi')
plot_results('../results_schwarz.csv', 'Schwarz')