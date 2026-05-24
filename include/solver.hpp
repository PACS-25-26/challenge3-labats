#pragma once
#include <cmath>
#include <vector>
#include <array>

namespace jacobisolver{

    using Point = std::array<double, 2>;
    using Grid = std::vector<Point>;

    class JacobiSolver{
        private:
        Grid grid;

    };

}