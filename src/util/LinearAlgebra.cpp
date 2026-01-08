#include "ag/util/LinearAlgebra.hpp"

#include <algorithm>
#include <cmath>

namespace ag::util {

std::vector<std::vector<double>>
computeGramMatrix(const std::vector<std::vector<double>>& X) {
    if (X.empty() || X[0].empty()) {
        return {};
    }

    const std::size_t n_obs = X.size();
    const std::size_t p = X[0].size();
    std::vector<std::vector<double>> XtX(p, std::vector<double>(p, 0.0));

    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t t = 0; t < n_obs; ++t) {
                XtX[i][j] += X[t][i] * X[t][j];
            }
        }
    }

    return XtX;
}

std::vector<double> computeXty(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y) {
    if (X.empty() || X[0].empty() || y.empty()) {
        return {};
    }

    const std::size_t n_obs = X.size();
    const std::size_t p = X[0].size();
    std::vector<double> Xty(p, 0.0);

    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t t = 0; t < n_obs; ++t) {
            Xty[i] += X[t][i] * y[t];
        }
    }

    return Xty;
}

std::vector<double> solveLinearSystem(std::vector<std::vector<double>>& A,
                                      std::vector<double>& b, double tol) {
    if (A.empty() || b.empty() || A.size() != b.size()) {
        return {};
    }

    const std::size_t n = A.size();

    // Gaussian elimination with partial pivoting
    for (std::size_t k = 0; k < n; ++k) {
        // Find pivot
        std::size_t max_row = k;
        double max_val = std::abs(A[k][k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > max_val) {
                max_val = std::abs(A[i][k]);
                max_row = i;
            }
        }

        // Swap rows if needed
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(b[k], b[max_row]);
        }

        // Check for singularity
        if (std::abs(A[k][k]) < tol) {
            return {};  // Singular matrix
        }

        // Forward elimination
        for (std::size_t i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (std::size_t j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    std::vector<double> x(n, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        if (std::abs(A[i][i]) < tol) {
            return {};  // Singular matrix
        }
        x[i] = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    return x;
}

std::vector<double> solveLeastSquares(const std::vector<std::vector<double>>& X,
                                      const std::vector<double>& y, double tol) {
    if (X.empty() || X[0].empty() || y.empty()) {
        return {};
    }

    // Compute X'X and X'y
    auto XtX = computeGramMatrix(X);
    auto Xty = computeXty(X, y);

    // Solve X'X * beta = X'y
    return solveLinearSystem(XtX, Xty, tol);
}

}  // namespace ag::util
