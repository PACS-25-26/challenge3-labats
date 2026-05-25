#include "solver.hpp"

namespace jacobisolver{

    const double& Grid::operator()(unsigned int i, unsigned int j) const{
        return U[i][j];
    }

    const double& Grid::operator()(unsigned int k) const{
        return U[k/n][k%n];
    }

    double& Grid::operator()(unsigned int i, unsigned int j){
        return U[i][j];
    }

    double& Grid::operator()(unsigned int k){
        return U[k/n][k%n];
    }

    const Point Grid::get_coordinate(unsigned int i, unsigned int j) const{
        return Point{i*h, j*h};
    }

    const Point Grid::get_coordinate(unsigned int k) const{
        return Point{(k/n) * h, (k%n) * h};
    }

    const double Grid::geth() const{
        return h;
    }

    const std::vector<std::vector<double>> Grid::getu() const{
        return U;
    }

    void JacobiSolver::set_boundary_cond(){
        for (auto j = 0; j < n; ++j){

            grid(0,j) = bordercondition(grid.get_coordinate(0,j));
            grid(n-1,j) = bordercondition(grid.get_coordinate(n-1,j));
            
            if (j != 0 && j != n-1){        //to not repeat evaluation at corners
                grid(j,0) = bordercondition(grid.get_coordinate(j,0));
                grid(j, n-1) = bordercondition(grid.get_coordinate(j, n-1));
            }

        }
    }

    std::vector<std::vector<double>> JacobiSolver::solve(){

        set_boundary_cond();
        bool converged = false;
        unsigned int k = 0;
        std::vector<std::vector<double>> uk = grid.getu();

        while (!converged && k <= maxit){

            k++;
            double difference = 0.;
            for (auto i = 1; i < n-1; ++i){
                for (auto j = 1; j < n-1; ++j){

                    double sum = 0.;
                    sum += uk[i-1][j] + uk[i+1][j] + uk[i][j-1] + uk[i][j+1];   //if i used grid i would do a mix of new and old grid elements; uk has only the old ones
                    sum += grid.geth() * grid.geth() * f(grid.get_coordinate(i,j));
                    grid(i, j) = 0.25 * sum;
                    difference += (grid(i, j) - uk[i][j])*(grid(i, j) - uk[i][j]);

                }
            }
            double increment = std::sqrt(grid.geth() * difference);
            if (increment < tolerance){
                converged = true;
                std::cout<<"Jacobi converged in "<<k<<" iterations!"<<std::endl;
            }
            uk = grid.getu();

        }

        if (!converged)
            std::cout<<"Jacobi did not converge."<<std::endl;

        return uk;

    }

}