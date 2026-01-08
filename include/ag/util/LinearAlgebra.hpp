/**
 * @file util/LinearAlgebra.hpp
 * @brief Common linear algebra utilities for least squares and matrix operations.
 *
 * This module provides reusable implementations of basic linear algebra operations
 * used throughout the codebase, particularly in statistical computations.
 */

#pragma once

#include <vector>

namespace ag::util {

/**
 * @brief Solve least squares problem: minimize ||Xβ - y||²
 *
 * Computes β̂ = (X'X)⁻¹X'y using Gaussian elimination with partial pivoting.
 * This implementation is suitable for small to moderate problem sizes (p < 100).
 *
 * @param X Design matrix (n_obs × p), stored row-major
 * @param y Response vector (n_obs)
 * @param tol Singularity tolerance for matrix operations
 * @return Estimated coefficients β̂ (size p), or empty vector if matrix is singular
 *
 * @note For large problems or better numerical stability, consider using
 *       QR decomposition or SVD-based methods.
 */
std::vector<double> solveLeastSquares(const std::vector<std::vector<double>>& X,
                                      const std::vector<double>& y, double tol = 1e-10);

/**
 * @brief Compute X'X (Gram matrix) for design matrix X.
 *
 * @param X Design matrix (n_obs × p), stored row-major
 * @return X'X matrix (p × p), stored as vector of vectors
 */
std::vector<std::vector<double>> computeGramMatrix(const std::vector<std::vector<double>>& X);

/**
 * @brief Compute X'y for design matrix X and response vector y.
 *
 * @param X Design matrix (n_obs × p), stored row-major
 * @param y Response vector (n_obs)
 * @return X'y vector (size p)
 */
std::vector<double> computeXty(const std::vector<std::vector<double>>& X,
                               const std::vector<double>& y);

/**
 * @brief Solve linear system Ax = b using Gaussian elimination with partial pivoting.
 *
 * Modifies A and b in-place during elimination.
 *
 * @param A Coefficient matrix (n × n), modified in-place
 * @param b Right-hand side vector (n), modified in-place
 * @param tol Singularity tolerance
 * @return Solution vector x (size n), or empty vector if matrix is singular
 */
std::vector<double> solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& b,
                                      double tol = 1e-10);

}  // namespace ag::util
