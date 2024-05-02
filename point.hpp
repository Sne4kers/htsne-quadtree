#pragma once
#include "math_utils.hpp"

struct Point {
    double x;
    double y;

    double sq_norm() {
        return x * x + y * y;
    }

    Point operator*(double a) {
        return Point{x * a, y * a};
    }

    Point operator/(double a) {
        return Point{x / a, y / a};
    }

    Point operator+(const Point& b) const {
        return Point{x + b.x, y + b.y};
    }

    Point to_klein() {
        double norm = sq_norm();
        return Point{hyperbolic_utils::poincare_to_klein(x, norm), hyperbolic_utils::poincare_to_klein(y, norm)};
    }

    Point to_poincare() {
        double norm = sq_norm();
        return Point{hyperbolic_utils::klein_to_poincare(x, norm), hyperbolic_utils::klein_to_poincare(y, norm)};
    }
};

        