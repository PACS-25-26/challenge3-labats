#include "solver.hpp"
using namespace jacobisolver;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned int n = 32; 

    unsigned int r = n % size;
    unsigned int local_size, rowfirst, rowfinal;
    if (rank < r){
        local_size = n / size + 1;
        rowfirst = rank * local_size;
        rowfinal = rowfirst + local_size - 1;
    }
    else{
        unsigned int displacement = rank - r;      //how many ranks before have had local_size = n/size
        local_size = n / size;
        rowfirst = r * (local_size + 1) + displacement * local_size;
        rowfinal = rowfirst + local_size - 1;
    }

    if (rank == 0)
        std::cout << "Grid: " << n << "x" << n << ", MPI ranks: " << size << std::endl;

    auto bc = [](const jacobisolver::Point& p) { return 0.0; };
    auto forcing = [](const jacobisolver::Point& p) {
        return 8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);
    };

    jacobisolver::JacobiSolver solver(n, rowfirst, rowfinal, forcing, bc);
    auto solution = solver.solve();

    unsigned int center_row = n / 2;
    unsigned int center_col = n / 2;

    if (center_row >= rowfirst && center_row <= rowfinal) 
        std::cout << "Rank " << rank << " holds the center. Central value: "<< solution[center_row - rowfirst][center_col] << std::endl;

    MPI_Finalize();
    return 0;
}