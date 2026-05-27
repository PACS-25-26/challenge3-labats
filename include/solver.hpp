#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <functional>
#include <iostream>
#include <mpi.h>

namespace jacobisolver{

    static const int ndim = 2;
    using Point = std::array<double, ndim>;
    
    class Grid {
    private:
        const unsigned int n;  // n_points for dimension, so total is n^2
        const double h;        // spacing
        const unsigned int first_row;   //first row of the matrix on which this grid works
        const unsigned int last_row;    //last row of the matrix on which this grid works
        std::vector<std::vector<double>> U;  // numerical solution matrix
        
    public:
        explicit Grid(unsigned int n_points, unsigned int row1, unsigned int rowlast) : 
            n(n_points), h(1.0/(n_points-1)), first_row(row1), last_row(rowlast), 
            U(rowlast - row1 + 1, std::vector<double>(n_points, 0.0)) {};

        double& operator()(unsigned int i, unsigned int j);
        double& operator()(unsigned int i);
        const double& operator()(unsigned int i, unsigned int j) const;
        const double& operator()(unsigned int k) const;

        const Point get_coordinate(unsigned int i, unsigned int j) const;
        const Point get_coordinate(unsigned int k) const;

        const double geth() const;
        const std::vector<std::vector<double>> getu() const;

};  

    static const std::function<double(const Point&)> zero = [](const Point& p) {return 0.;};

    class JacobiSolver{
        private:
        unsigned int n;         
        unsigned int first_row;
        unsigned int last_row;
        Grid grid;
        std::function<double(const Point&)> f;
        std::function<double(const Point&)> bordercondition;
        static constexpr double tolerance = 1e-4;
        static constexpr unsigned int maxit = 2000;

        public:
        explicit JacobiSolver(unsigned int np, unsigned int row1, unsigned int rowlast, 
            std::function<double(const Point&)> func, std::function<double(const Point&)> bc = zero) : 
            n(np), first_row(row1), last_row(rowlast), grid(np, row1, rowlast), f(func), bordercondition(bc) {};

        void set_boundary_cond();

        std::vector<std::vector<double>> solve();

    };
}