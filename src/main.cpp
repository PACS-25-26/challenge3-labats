#include <mpi.h>
#include "solver.hpp"
using namespace jacobisolver;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        // Test sul processo 0 per ora (senza parallelismo vero)
        
        unsigned int n = 64;  // griglia 16x16
        
        // Boundary condition: u = 0 ovunque
        auto bc = [](const Point& p) { return 0.0; };
        
        // Forcing term: f(x,y) = 8π² sin(2πx) sin(2πy)
        auto forcing = [](const Point& p) {
            double x = p[0], y = p[1];
            return 8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
        };
        
        JacobiSolver solver(n, forcing, bc);
        auto solution = solver.solve();
        
        std::cout << "Solution computed. Grid size: " << n << "x" << n << std::endl;
        std::cout << "Central point value: " << solution[n/2][n/2] << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}