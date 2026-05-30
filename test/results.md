# Performance Notes and Analysis

When the benchmark tests to evaluate the computation times and convergence rates were executed on the cluster (grid sizes from 16 to 256, doubling each time, using 1, 2, and 4 MPI ranks), the results highlighted several key behaviors of distributed linear solvers.

## Standard Jacobi: Super-linear Speedup and Premature Convergence
The pure iterative Jacobi method showcased exceptional scalability. Notably, for large grids (n=256), a super-linear speedup was observed:
- 1 Rank: 32.9 seconds
- 4 Ranks: 5.3 seconds

However, the L2 error did not scale as optimally for finer meshes. While the error decreased regularly as the grid size grew up to n=64, it began to increase again for n=128 and n=256. This behavior highlights an intrinsic limitation of the Jacobi method. For very fine grids, the difference between consecutive iteration steps becomes extremely small. Consequently, the algorithm reaches the tolerance threshold (`increment < tol`) and halts prematurely, falsely identifying convergence long before reaching the true analytical solution.

## Domain Decomposition: Schwarz LU Overhead
The Additive Schwarz method, which leverages `Eigen::SparseLU` for exact internal block solves, demonstrated the physics of domain propagation:
- Single-Core: With 1 rank, the entire global matrix is factored once. The solver converges in exactly 2 iterations, yielding extremely fast execution times.
- Multi-Core: When split across multiple ranks with no overlap, communication is restricted to the boundary rows. Consequently, the number of iterations scales linearly with the grid size as information takes O(n) steps to cross the global domain (e.g., 410 iterations for N=256 on 4 ranks).

In terms of execution time, the parallel Schwarz method suffered a performance penalty compared to the single-core exact solve. Resolving a direct LU substitution hundreds of times inside the iterative while loop overshadowed the computational benefits of factorizing smaller sub-matrices.

The Additive Schwarz solver successfully converged to the exact analytical solutions. The L2 error analysis demonstrates a perfect O(h^2) spatial convergence rate, perfectly matching the theoretical expectations for a second-order central finite difference scheme. 

Testing the method with non-homogeneous Dirichlet boundary conditions yielded again optimal results, showing that the methods both converged to the numerical exact solution.