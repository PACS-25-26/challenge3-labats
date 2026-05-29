# Challenge 3
The third challenge

The code provides an implementation of a parallel solver for the 2D Laplace/Poisson equation with Dirichlet boundary conditions. It supports two different numerical approaches to solve the resulting sparse linear system:
1. **Standard Point-Jacobi:** A classic iterative relaxation scheme.
2. **Schwarz-type (Block Jacobi):** A domain decomposition approach where each MPI rank solves its internal block exactly using Eigen's Sparse LU Factorization.

The implementation is parallelized using MPI for distributed-memory data exchanges and OpenMP for shared-memory "for" loop accelerations.

For the homogeneous problem, the forcing term is set to $f(x) = 8 π^2 \sin(2πx) \sin(2πy)$, and the exact solution is $u(x, y) = \sin(2πx) \sin(2πy)$.

For the inhomogeneous problem, the forcing term is set to 0, and the border condition is $g(x,y) = e^x \cos(y)$, with exact solution $u(x,y) = e^x \cos(y)$

In `main.cpp` these given functions can be changed manually.

## Compilation and Execution
In the directory, simply type `make` to produce the executable, then, since the code relies on MPI, you must run it using `mpirun`. The executable accepts up to three optional command-line arguments:

```bash
mpirun -np <ranks> ./jacobi [grid_size] [solver_type] [bc_type]
```

- grid_size: the number of nodes along one dimension (e.g., 32, 64, 256). Defaults to 32.
- solver_type: 0 for Standard Jacobi, 1 for Schwarz LU. Defaults to 0.
- bc__type: 0 for homogeneous (zero) boundary conditions (sinusoidal forcing), 1 for nonhomogeneous boundary conditions (Laplace problem). Defaults to 0.

Example:
```bash
mpirun -np 4 ./jacobi 128 1 0
```
This will run the code with 4 ranks, on a grid that has 128 on each dimension (for a total of 16,384 nodes), using the Schwarz LU solver, with homogeneous border conditions.

## The `Grid` Class
This class manages the spatial domain and memory allocation for the solver. It stores the global grid size `n`, the indices of the first and last rows assigned to the local MPI rank, and the local grid nodes. 

In a multi-rank MPI environment, each rank creates a smaller `Grid` representing a horizontal slice of the global domain. The number of rows is distributed as evenly as possible across all available ranks by the `main.cpp`. 

The class provides getter methods and overloads the `()` operator for element access to the local grid `U` (implemented as a `std::vector<std::vector<double>>`). Finally, the `swap_u` method performs a `std::swap` between the current local grid and a newly computed iteration matrix. This swap is efficient and crucial for the optimal memory performance of the iterative `solve` algorithm.

## The `JacobiSolver` Class
This class encapsulates the core numerical algorithms required to solve the Poisson equation. It advances the selected iterative algorithm until the system reaches convergence or hits a predefined iteration limit to prevent infinite loops. By default, the maximum number of iterations (`maxit`) is set to 50,000, and the `tolerance` on the local increment is set to $10^{-5}$.

It provides two distinct methods to compute the solution:
* `solve()`: Executes the standard distributed Jacobi iterative method.
* `solve_schwarz()`: Implements an Additive Schwarz (Block Jacobi) domain decomposition, utilizing Eigen's Sparse LU factorization for exact local block solves.

## Structure of the directory
In the directory `include`, the header `solver.hpp` declares the `Grid` and `JacobiSolver` classes, in the broader `jacobisolver` namespace that also includes the VTK format export utilities.
In `src`, `solver.cpp` implements all the methods of the classes hadling MPI communications and the VTK export, while `main.cpp` handles MPI initialization, the problem setup and gathers the final results in rank 0.
The directory `test` contains the PBS script `run_scalability.pbs` required to submit the benchmark in the HPC cluster that was used for testing and execution. The two output CSV files will be generated here, one for each numerical solving method.
In `test/data`, the Python script `plot_results.py` is used to parse the CSV files and plot the scalability and L2 error graphs.

## Cluster Execution & Post-Processing
To evaluate the performance of both algorithms, one can submit the benchmark job to the cluster's queue:
```bash
cd test
qsub run_all.pbs
```
Once the job is completed, it will generate the result CSV files in the `test` directory. To visualize the performance speedup and the L2 error convergence, navigate to the `data` folder and run the Python script:
```bash
cd data
python plot_results.py
```
This will automatically generate the `scalability_<solver>.png` and `error_L2_<solver>.png charts`.
