#include "solver.hpp"
using namespace jacobisolver;

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try{

        if (argc > 4){      //Exception handling
            if (rank == 0) 
                std::cerr << "Too many arguments in input. Enter only: n, solver_type, bc." << std::endl;
            MPI_Finalize();
            return 1;
        }

        unsigned int n = 32; // Default value
        int solver_type = 0; //Classic Jacobi or Schwarz-type
        int bc_type = 0; //Homogeneous BC if 0, inhomogeneous if 1
        if (argc > 1) {
            n = std::atoi(argv[1]);
        } 
        if (argc > 2) {
            solver_type = std::atoi(argv[2]);
        }
        if (argc > 3) {
            bc_type = std::atoi(argv[3]);
        }
        double h = 1./(n-1);

        if (size > n){      //Exception handling
            if (rank == 0) 
                std::cerr << "Too many MPI ranks for grid size." << n << std::endl;
            MPI_Finalize();
            return 1;
        }

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

        std::string s_name = (solver_type == 1) ? "Schwarz LU" : "Standard Jacobi";
        if (rank == 0) 
            std::cout << "Grid: " << n << "x" << n << ", MPI ranks: " << size << " Solver: " << s_name << std::endl;

        std::function<double(const Point&)> bc;
        std::function<double(const Point&)> forcing;
        std::function<double(const Point&)> u_exact;

        if (bc_type == 0) {        //bc is default 0
            bc = [](const Point&){ return 0.0; };
            forcing = [](const Point& p) {
                return 8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);
            };
            u_exact = [](const Point& p){
                return std::sin(2.0 * M_PI * p[0]) * std::sin(2.0 * M_PI * p[1]);
            };
        } 
        else {        //u = e^x * cos(y), forcing = 0, bc not 0
            bc = [](const Point& p) { return std::exp(p[0]) * std::cos(p[1]); };
            forcing = [](const Point&) { return 0.0; };
            u_exact = [](const Point& p){ return std::exp(p[0]) * std::cos(p[1]); };
        }

        JacobiSolver solver(n, rowfirst, rowfinal, forcing, bc);

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        std::vector<std::vector<double>> solution;

        if (solver_type == 1) {
            solution = solver.solve_schwarz();
        } else {
            solution = solver.solve();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();

        double elapsed_time = t_end - t_start;

        double l2norm = 0, l2final = 0;
        #pragma omp parallel for reduction(+: l2norm)
        for (auto i = rowfirst; i <= rowfinal; ++i){
            for (auto j = 0u; j < n; ++j){

                Point coordinate = {i*h, j*h};
                l2norm += (solution[i - rowfirst][j] - u_exact(coordinate))*(solution[i - rowfirst][j] - u_exact(coordinate));

            }
        }

        MPI_Reduce(&l2norm, &l2final, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        std::vector<double> finalgrid;      //MPI works mainly with contiguous memory, so I prefer to use a "flattened" std::vector<double>

        std::vector<double> solution_flat(local_size * n);
        #pragma omp parallel for
        for (unsigned int i = 0; i < local_size; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                solution_flat[i * n + j] = solution[i][j];
            }
        }

        if (rank == 0)
            finalgrid.resize(n*n, 0.);      //The total solution is gathered only by rank 0

        MPI_Gatherv(solution_flat.data(), local_size * n, MPI_DOUBLE, finalgrid.data(), sizes.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0){
            
            l2final *= h;
            l2final = std::sqrt(l2final);

            if (bc_type == 0) {        //csv format for graphs
                std::cout << n << "," << size << "," << elapsed_time << "," << l2final << std::endl;
                if (n == 256) export_vtk(n, finalgrid, h);
            } 
            else {      //for nonhomogeneous bc we simply print the results
                std::cout << "Nonhomogeneous bc.  Grid: " << n << "x" << n << " Ranks: " << size << " " << s_name << " L2 Error: " << l2final << std::endl;
            }

        }
    }
    catch (const std::exception& e){           //tries to catch the memory overloads or fatal errors
        std::cerr << "Process " << rank << " caught exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}