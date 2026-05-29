#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <functional>
#include <exception>
#include <iostream>
#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace jacobisolver{

    static const int ndim = 2;
    using Point = std::array<double, ndim>;
    
    class Grid {
    private:
        unsigned int n;  // n_points for dimension, so total is n^2
        double h;        // spacing
        unsigned int first_row;   //first row of the matrix on which this grid works
        unsigned int last_row;    //last row of the matrix on which this grid works
        std::vector<std::vector<double>> U;  // numerical solution matrix
        
    public:
        explicit Grid(unsigned int n_points, unsigned int row1, unsigned int rowlast) : 
            n(n_points), h(1.0/(n_points-1)), first_row(row1), last_row(rowlast), 
            U(rowlast - row1 + 1, std::vector<double>(n_points, 0.0)) {};

        double& operator()(unsigned int i, unsigned int j) noexcept;
        double& operator()(unsigned int i) noexcept;
        const double& operator()(unsigned int i, unsigned int j) const noexcept;
        const double& operator()(unsigned int k) const noexcept;

        Point get_coordinate(unsigned int i, unsigned int j) const noexcept;
        Point get_coordinate(unsigned int k) const noexcept;

        double geth() const noexcept;
        const std::vector<std::vector<double>>& getu() const noexcept;
        void swap_u(std::vector<std::vector<double>>& other) noexcept;

};  

    //std::function<double(const Point&)> zero = [](const Point& p) {return 0.0;};

    class JacobiSolver{
        private:
        unsigned int n;         
        unsigned int first_row;
        unsigned int last_row;
        Grid grid;
        std::function<double(const Point&)> f;
        std::function<double(const Point&)> bordercondition;
        static constexpr double tolerance = 1e-5;
        static constexpr unsigned int maxit = 50000;

        public:
        explicit JacobiSolver(unsigned int np, unsigned int row1, unsigned int rowlast, 
            std::function<double(const Point&)> func, std::function<double(const Point&)> bc = [](const Point& p){ return 0.0; }) : 
            n(np), first_row(row1), last_row(rowlast), grid(np, row1, rowlast), f(func), bordercondition(bc) {};

        void set_boundary_cond() noexcept;

        std::vector<std::vector<double>> solve();

        std::vector<std::vector<double>> solve_schwarz();

    };

    void export_vtk(unsigned int n, const std::vector<double>& solution, double h);

}