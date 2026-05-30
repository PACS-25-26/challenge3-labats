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

/**
 * @namespace jacobisolver
 * @brief Namespace containing all the tools to solve the Laplace/Poisson equation using parallel methods.
 */
namespace jacobisolver{

    static const int ndim = 2;

    /** @brief Alias for a 2D coordinate point. */
    using Point = std::array<double, ndim>;
    
    /**
     * @class Grid
     * @brief Manages the 2D grid structure, memory allocation, and coordinate mapping for the local MPI rank.
     */
    class Grid {
    private:
        unsigned int n;  // n_points for dimension, so total is n^2
        double h;        // spacing
        unsigned int first_row;   //first row of the matrix on which this grid works
        unsigned int last_row;    //last row of the matrix on which this grid works
        std::vector<std::vector<double>> U;  // numerical solution matrix
        
    public:
        /**
         * @brief Constructor for the Grid class.
         * @param n_points Global grid dimension (n x n).
         * @param row1 Global index of the first local row.
         * @param rowlast Global index of the last local row.
         */
        explicit Grid(unsigned int n_points, unsigned int row1, unsigned int rowlast) : 
            n(n_points), h(1.0/(n_points-1)), first_row(row1), last_row(rowlast), 
            U(rowlast - row1 + 1, std::vector<double>(n_points, 0.0)) {};

        double& operator()(unsigned int i, unsigned int j) noexcept;
        double& operator()(unsigned int i) noexcept;
        const double& operator()(unsigned int i, unsigned int j) const noexcept;
        const double& operator()(unsigned int k) const noexcept;

        /**
         * @brief Computes the physical coordinates of a given global grid node.
         * @param i Global row index.
         * @param j Global column index.
         * @return A Point representing the (x, y) coordinates in the domain [0,1]x[0,1].
         */
        Point get_coordinate(unsigned int i, unsigned int j) const noexcept;
        /**
         * @brief Computes the physical coordinates of a given global numbered grid node.
         * @param k Global element index. k = i * n + j.
         * @return A Point (std::array of 2 doubles) representing the (x, y) coordinates in the domain [0,1]x[0,1].
         */
        Point get_coordinate(unsigned int k) const noexcept;

        /**
         * @brief Returns the grid spacing h.
         * @return The uniform distance between adjacent points in the grid.
         */
        double geth() const noexcept;

        /**
         * @brief Returns a constant reference to the local grid matrix.
         * @return Const reference to a 2D std::vector containing the local solution elements.
         */
        const std::vector<std::vector<double>>& getu() const noexcept;

        /**
         * @brief Swaps the current grid data with a new matrix (in the solver it is used with the new computed grid).
         * * @param other The updated grid values to be swapped in.
         */
        void swap_u(std::vector<std::vector<double>>& other) noexcept;

};  

    /**
     * @class JacobiSolver
     * @brief Solves the Laplace/Poisson equation using iterative methods.
     * * Uses a Grid to manage the physical domain directly.
     */
    class JacobiSolver{
        private:
        unsigned int n;             // n_points for dimension, so total is n^2
        unsigned int first_row;     //first row of the matrix on which this grid works
        unsigned int last_row;      //last row of the matrix on which this grid works
        Grid grid;                  //the local grid on which the rank operates
        std::function<double(const Point&)> f;      //the forcing function
        std::function<double(const Point&)> bordercondition;        //the Dirichlet boundary condition function
        static constexpr double tolerance = 1e-5;                   //the tolerance of the solver
        static constexpr unsigned int maxit = 50000;                //the maximum consented number of iterations

        public:
        /**
         * @brief Constructor for the JacobiSolver.
         * @param np Global grid size.
         * @param row1 Global index of the first local row.
         * @param rowlast Global index of the last local row.
         * @param func Function representing the forcing term of the PDE.
         * @param bc Function representing the boundary conditions of the PDE.
         */
        explicit JacobiSolver(unsigned int np, unsigned int row1, unsigned int rowlast, 
            std::function<double(const Point&)> func, std::function<double(const Point&)> bc = [](const Point& p){ return 0.0; }) : 
            n(np), first_row(row1), last_row(rowlast), grid(np, row1, rowlast), f(func), bordercondition(bc) {};

        /**
         * @brief Applies the boundary conditions to the local grid.
         * * Evaluates the function 'bc' only the physical boundaries of the global domain.
         * OpenMP directives are used to speedup for loops.
         */
        void set_boundary_cond() noexcept;

        /**
         * @brief Solves the system using the standard iterative Point-Jacobi method.
         * * This method exchanges ghost rows via MPI and accelerates internal loops via OpenMP. It also checks whether the first
         * row is 0, thus meaning that it is imposed by border conditions and that it cannot be swapped with the previous rank, and
         * does the same for if the last row is n-1 (no following row). 
         * OpenMP directives are used to speedup for loops.
         * * @return A 2D vector containing the converged local portion of the solution grid.
         */
        std::vector<std::vector<double>> solve();

        /**
         * @brief Solves the system using an Additive Schwarz Domain Decomposition method.
         * * Assembles the local finite difference matrix using Eigen::SparseMatrix and solves the internal block exactly 
         * using Eigen's Sparse LU factorization, iterating only at the MPI interfaces. OpenMP directives are used to speedup
         * for loops.
         * * @return A 2D vector containing the converged local portion of the solution grid.
         */
        std::vector<std::vector<double>> solve_schwarz();

    };

    /**
     * @brief Exports the global solution to a VTK format for ParaView visualization.
     * @param n Global grid dimension.
     * @param solution Flattened 1D vector containing the gathered global solution.
     * @param h Grid spacing.
     */
    void export_vtk(unsigned int n, const std::vector<double>& solution, double h);

}