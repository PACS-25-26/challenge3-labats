#include "solver.hpp"
#include <omp.h>
#include <fstream>

namespace jacobisolver{

    const double& Grid::operator()(unsigned int i, unsigned int j) const{
        if (i < first_row || i > last_row || j >= n)
            throw std::out_of_range("Index out of bounds");
        return U[i - first_row][j];
    }

    const double& Grid::operator()(unsigned int k) const{
        unsigned int i = k/n, j = k%n;
        if (i < first_row || i > last_row || j >= n)
            throw std::out_of_range("Index out of bounds");
        return U[i - first_row][j];
    }

    double& Grid::operator()(unsigned int i, unsigned int j){
        if (i < first_row || i > last_row || j >= n)
            throw std::out_of_range("Index out of bounds");
        return U[i - first_row][j];
    }

    double& Grid::operator()(unsigned int k){
        unsigned int i = k/n, j = k%n;
        if (i < first_row || i > last_row || j >= n)
            throw std::out_of_range("Index out of bounds");
        return U[i - first_row][j];
    }

    const Point Grid::get_coordinate(unsigned int i, unsigned int j) const{
        if (i < first_row || i > last_row || j >= n)
            throw std::out_of_range("Index out of bounds");
        return Point{i*h, j*h};
    }

    const Point Grid::get_coordinate(unsigned int k) const{
        unsigned int i = k/n, j = k%n;
        if (i < first_row || i > last_row || j >= n)
            throw std::out_of_range("Index out of bounds");
        return Point{i*h, j*h};
    }

    const double Grid::geth() const{
        return h;
    }

    const std::vector<std::vector<double>> Grid::getu() const{
        return U;
    }

    void JacobiSolver::set_boundary_cond(){
        if (first_row == 0){        //only for the first rank

            for (auto j = 0; j < n; ++j)
                grid(0, j) = bordercondition(grid.get_coordinate(0,j));

        }
        if (last_row == n - 1){     //only for the last rank

            for (auto j = 0; j < n; ++j)
                grid(n-1, j) = bordercondition(grid.get_coordinate(n-1,j));

        }
        for (auto i = first_row; i <= last_row; ++i){
            if (i != 0 && i != n-1){  // already set before

                grid(i,0) = bordercondition(grid.get_coordinate(i,0));
                grid(i, n-1) = bordercondition(grid.get_coordinate(i, n-1));

            }
        }
        }

    std::vector<std::vector<double>> JacobiSolver::solve(){

        int rank,size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        set_boundary_cond();

        int local_converged = 0, global_converged = 0;
        unsigned int k = 0;
        std::vector<std::vector<double>> local_uk = grid.getu();

        while (!global_converged && k <= maxit){

            k++;
            local_converged = 0;
            double difference = 0.;
            std::vector<double> followingrow(n, 0.);
            std::vector<double> previousrow(n, 0.);

            if (last_row != n-1 && first_row != 0){
                //indices of local_uk go from 0 to local_size - 1
                //the last index of local_uk is lastrow - firstrow
                //std::cout<<"I'm rank "<<rank<<" and I'm entering the first block"<<std::endl;
                MPI_Sendrecv(local_uk[0].data(), n, MPI_DOUBLE, 
                            rank - 1, 0, followingrow.data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(local_uk[last_row - first_row].data(), n, MPI_DOUBLE, 
                            rank + 1, 0, previousrow.data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //std::cout<<"I'm rank "<<rank<<" and I'm exiting the first block"<<std::endl;
            }
            if (first_row == 0 && rank + 1 < size){     //we need to check that a later rank does exist
                //std::cout<<"I'm rank "<<rank<<" and I'm entering the second block"<<std::endl;
                MPI_Sendrecv(local_uk[last_row - first_row].data(), n, MPI_DOUBLE, 
                            rank + 1, 0, followingrow.data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //std::cout<<"I'm rank "<<rank<<" and I'm exiting the second block"<<std::endl;
            }
            if (last_row == n - 1 && rank - 1 >= 0){    //we need to check that a previous rank does exist
                //std::cout<<"I'm rank "<<rank<<" and I'm entering the third block"<<std::endl;
                MPI_Sendrecv(local_uk[0].data(), n, MPI_DOUBLE, 
                            rank - 1, 0, previousrow.data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //std::cout<<"I'm rank "<<rank<<" and I'm exiting the third block"<<std::endl;
            }

            if (first_row != last_row){
                #pragma omp parallel for reduction(+: difference)
                for (auto j = 1; j < n-1; ++j){
                    if (first_row != 0){
                    
                        //first row update using previousrow, assuming that first_row != last_row
                        double sum = 0.;
                        //indices of local_uk go from 0 to local_size - 1
                        sum += previousrow[j] + local_uk[1][j] + local_uk[0][j-1] + local_uk[0][j+1];   //if i used grid i would do a mix of new and old grid elements; uk has only the old ones
                        sum += grid.geth() * grid.geth() * f(grid.get_coordinate(first_row,j));
                        grid(first_row, j) = 0.25 * sum;
                        difference += (grid(first_row, j) - local_uk[0][j])*(grid(first_row, j) - local_uk[0][j]);

                    }

                    if (last_row != n-1){

                        //last row update using followingrow, assuming that first_row != last_row
                        double sum = 0.;
                        sum += local_uk[last_row - first_row - 1][j] + followingrow[j] + local_uk[last_row - first_row][j-1] + local_uk[last_row - first_row][j+1];   //if i used grid i would do a mix of new and old grid elements; uk has only the old ones
                        sum += grid.geth() * grid.geth() * f(grid.get_coordinate(last_row,j));
                        grid(last_row, j) = 0.25 * sum;
                        difference += (grid(last_row, j) - local_uk[last_row - first_row][j])*(grid(last_row, j) - local_uk[last_row - first_row][j]);

                    }
                }
            }
            #pragma omp parallel for reduction(+:difference)
            for (auto i = first_row + 1; i < last_row; ++i){
               for (auto j = 1; j < n-1; ++j){

                    double sum = 0.;
                    //again, all indices of local_uk must be shifted by a firstrow since they start from 0
                    sum += local_uk[i-1 - first_row][j] + local_uk[i+1 - first_row][j] + local_uk[i - first_row][j-1] + local_uk[i - first_row][j+1];   //if i used grid I would do a mix of new and old grid elements; uk has only the old ones
                    sum += grid.geth() * grid.geth() * f(grid.get_coordinate(i,j));
                    grid(i, j) = 0.25 * sum;
                    difference += (grid(i, j) - local_uk[i - first_row][j])*(grid(i, j) - local_uk[i - first_row][j]);

                }
            }

            if (first_row == last_row){
                if (first_row != 0 && last_row != n-1){
                    #pragma omp parallel for reduction(+:difference)
                    for (auto j = 1; j < n-1; ++j){

                        double sum = 0.;
                        sum += previousrow[j] + followingrow[j] + local_uk[0][j-1] + local_uk[0][j+1];   //if i used grid I would do a mix of new and old grid elements; uk has only the old ones
                        sum += grid.geth() * grid.geth() * f(grid.get_coordinate(first_row,j));
                        grid(first_row, j) = 0.25 * sum;
                        difference += (grid(first_row, j) - local_uk[0][j])*(grid(first_row, j) - local_uk[0][j]);

                }
                }
            }

            double increment = std::sqrt(grid.geth() * difference);
            if (increment < tolerance)
                local_converged = 1;


            MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

            if (global_converged && rank == 0)
                std::cout<<"Jacobi converged in "<<k<<" iterations!"<<std::endl;
            local_uk = grid.getu();

        }

        if (!global_converged && rank == 0)
            std::cout<<"Jacobi did not converge."<<std::endl;

        return local_uk;

    }

    void export_vtk(unsigned int n, std::vector<double> solution, double h){

        std::ofstream file("jacobi_solution.vtk");

        // vtk header has always the same format
        file << "# vtk DataFile Version 3.0\n";
        file << "Laplace Solution with Jacobi\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << n << " " << n << " 1\n";       //must be 3d, so z dimension is 1
        file << "ORIGIN 0.0 0.0 0.0\n";
        file << "SPACING " << h << " " << h << " 1.0\n";
        
        // Data details
        int num_points = n * n;
        file << "POINT_DATA " << num_points << "\n";
        file << "SCALARS solution double 1\n";
        file << "LOOKUP_TABLE default\n";

        for (auto i : solution)
            file << i << std::endl;

        file.close();
        std::cout<<"Vtk file created successfully!"<<std::endl;

    }

}
