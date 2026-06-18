#include "ag/util/LinearAlgebra.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ag::util {

std::vector<std::vector<double>> computeGramMatrix(const std::vector<std::vector<double>>& X) {
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

std::vector<double> solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& b,
                                      double tol) {
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

double olsTStatistic(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                     std::size_t coef_index, double tol) {
    if (y.empty() || X.empty() || X[0].empty()) {
        throw std::invalid_argument("olsTStatistic: empty design matrix or response");
    }

    const std::size_t n = y.size();
    const std::size_t k = X[0].size();

    if (coef_index >= k) {
        throw std::invalid_argument("olsTStatistic: coefficient index out of range");
    }
    if (n <= k) {
        throw std::invalid_argument("olsTStatistic: insufficient degrees of freedom");
    }

    std::vector<double> beta = solveLeastSquares(X, y, tol);
    if (beta.empty()) {
        throw std::runtime_error("olsTStatistic: singular design matrix");
    }

    // Residual sum of squares.
    double rss = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        double fitted = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            fitted += X[t][i] * beta[i];
        }
        const double resid = y[t] - fitted;
        rss += resid * resid;
    }
    const double sigma2 = rss / static_cast<double>(n - k);

    // Diagonal element of (X'X)^{-1} for coef_index: solve X'X v = e_i.
    std::vector<double> ei(k, 0.0);
    ei[coef_index] = 1.0;
    auto XtX = computeGramMatrix(X);
    std::vector<double> inv_col = solveLinearSystem(XtX, ei, tol);
    if (inv_col.empty()) {
        throw std::runtime_error("olsTStatistic: failed to compute standard errors");
    }

    const double se = std::sqrt(sigma2 * inv_col[coef_index]);
    return beta[coef_index] / se;
}

}  // namespace ag::util
