#pragma once
#include <cmath>

namespace hyperbolic_utils {
    double sq_norm(const double& a, const double& b) {
        return a * a + b * b;
    }

    double poincare_to_klein(const double& a, const double& sq_n) {
        return 2 * a / (1 + sq_n);
    }

    double klein_to_poincare(const double& a, const double& sq_n) {
        return a / (1 + std::sqrt(1 - sq_n));
    }

    double lorentz_factor(const double& sq_n) {
        return 1 / std::sqrt(1 - sq_n);
    }
}