#include "solver.hpp"
using namespace jacobisolver;

int main(int argc, char* argv[]) {
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::cout << "Warning: MPI non supporta il threading richiesto!" << std::endl;
    }
    

    unsigned int n = 32; 

    unsigned int r = n % size;
    unsigned int local_size, rowfirst, rowfinal;
    std::vector<int> sizes(size);
    std::vector<int> displacements(size);
    int offset = 0;

    for (auto i = 0; i < size; ++i){
        sizes[i] = (i < r) ? (n/size + 1) * n : (n/size) * n;
        displacements[i] = offset;
        offset += sizes[i];
    }

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
    std::vector<std::vector<double>> solution = solver.solve();

    std::vector<double> finalgrid;          //MPI works only with contiguous memory, so I can't use vector<vector> like before

    if (rank == 0)
        finalgrid.resize(n*n, 0.);      //The total solution is gathered only by rank 0

    MPI_Gatherv(solution.data(), local_size * n, MPI_DOUBLE, finalgrid.data(), sizes.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
        export_vtk(n, finalgrid, 1.0/n);


    unsigned int center_row = n / 2;
    unsigned int center_col = n / 2;

    if (center_row >= rowfirst && center_row <= rowfinal) 
        std::cout << "Rank " << rank << " holds the center. Central value: "<< solution[center_row - rowfirst][center_col] << std::endl;

    MPI_Finalize();
    return 0;
}