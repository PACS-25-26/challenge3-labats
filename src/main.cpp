#include "solver.hpp"
using namespace jacobisolver;

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned int n = 32; // Default value
    if (argc > 1) {
        n = std::atoi(argv[1]);
    } 
    double h = 1./(n-1);

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

    auto bc = [](const Point& p) { return 0.0; };
    auto forcing = [](const Point& p) {
        return 8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);
    };
    auto u_exact = [](const Point& p){
        return std::sin(2.0 * M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);
    };

    JacobiSolver solver(n, rowfirst, rowfinal, forcing, bc);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    std::vector<std::vector<double>> solution = solver.solve_schwarz();

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    double elapsed_time = t_end - t_start;

    double l2norm = 0, l2final = 0;
    for (auto i = rowfirst; i <= rowfinal; ++i){
        for (auto j = 0; j < n; ++j){

            Point coordinate = {i*h, j*h};
            l2norm += (solution[i - rowfirst][j] - u_exact(coordinate))*(solution[i - rowfirst][j] - u_exact(coordinate));

        }
    }

    MPI_Reduce(&l2norm, &l2final, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<double> finalgrid;      //MPI works only with contiguous memory, so I can't use vector<vector> like before

    if (rank == 0)
        finalgrid.resize(n*n, 0.);      //The total solution is gathered only by rank 0

    MPI_Gatherv(solution.data(), local_size * n, MPI_DOUBLE, finalgrid.data(), sizes.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){
        
        l2final *= h;
        l2final = std::sqrt(l2final);
        std::cout << n << "," << size << "," << elapsed_time << "," << l2final << std::endl;
        if (n == 256)
            export_vtk(n, finalgrid, h);

    }

    MPI_Finalize();
    return 0;
}