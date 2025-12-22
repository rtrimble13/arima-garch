#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ag::estimation {

/**
 * @brief Vector of parameter values with bounds checking.
 *
 * ParameterVector is a thin wrapper around std::vector<double> providing
 * a type-safe container for model parameters. It is serialization-friendly
 * as it contains no raw pointers and uses standard library containers.
 *
 * This class represents a sequence of numeric parameter values that can be
 * used for optimization, estimation, or simulation.
 */
class ParameterVector {
public:
    /**
     * @brief Default constructor creates an empty parameter vector.
     */
    ParameterVector() = default;

    /**
     * @brief Construct a parameter vector with specified size.
     * @param size Number of parameters
     * @param initial_value Initial value for all parameters (default: 0.0)
     */
    explicit ParameterVector(std::size_t size, double initial_value = 0.0)
        : values_(size, initial_value) {}

    /**
     * @brief Construct from a vector of values.
     * @param values Vector of parameter values
     */
    explicit ParameterVector(const std::vector<double>& values) : values_(values) {}

    /**
     * @brief Construct from a vector of values (move semantics).
     * @param values Vector of parameter values
     */
    explicit ParameterVector(std::vector<double>&& values) : values_(std::move(values)) {}

    /**
     * @brief Get the number of parameters.
     * @return Size of the parameter vector
     */
    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

    /**
     * @brief Check if the parameter vector is empty.
     * @return true if size is 0, false otherwise
     */
    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }

    /**
     * @brief Access parameter value by index (const).
     * @param index Parameter index
     * @return Parameter value at index
     * @throws std::out_of_range if index >= size()
     */
    [[nodiscard]] double operator[](std::size_t index) const { return values_.at(index); }

    /**
     * @brief Access parameter value by index (non-const).
     * @param index Parameter index
     * @return Reference to parameter value at index
     * @throws std::out_of_range if index >= size()
     */
    double& operator[](std::size_t index) { return values_.at(index); }

    /**
     * @brief Get const reference to underlying vector.
     * @return Const reference to internal std::vector<double>
     */
    [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

    /**
     * @brief Get mutable reference to underlying vector.
     * @return Reference to internal std::vector<double>
     */
    std::vector<double>& values() noexcept { return values_; }

    /**
     * @brief Resize the parameter vector.
     * @param new_size New size for the vector
     * @param value Value to initialize new elements (default: 0.0)
     */
    void resize(std::size_t new_size, double value = 0.0) { values_.resize(new_size, value); }

    /**
     * @brief Clear all parameters.
     */
    void clear() noexcept { values_.clear(); }

private:
    std::vector<double> values_;  // Serialization-friendly: no raw pointers
};

/**
 * @brief Container for ARIMA-GARCH model parameters.
 *
 * ModelParameters separates ARIMA and GARCH parameters and provides
 * structured access to model coefficients. This is distinct from model
 * specifications (ArimaSpec, GarchSpec) which define model structure.
 *
 * The parameter layout follows standard ARIMA-GARCH notation:
 * - ARIMA section: intercept/mean, AR coefficients (φ), MA coefficients (θ)
 * - GARCH section: omega (ω), ARCH coefficients (α), GARCH coefficients (β)
 *
 * This class is serialization-friendly as it uses standard library containers
 * without raw pointers.
 */
class ModelParameters {
public:
    /**
     * @brief Default constructor creates empty parameters.
     */
    ModelParameters() = default;

    /**
     * @brief Construct with separate ARIMA and GARCH parameters.
     * @param arima_params ARIMA parameter vector
     * @param garch_params GARCH parameter vector
     */
    ModelParameters(ParameterVector arima_params, ParameterVector garch_params)
        : arima_params_(std::move(arima_params)), garch_params_(std::move(garch_params)) {}

    /**
     * @brief Construct with specified sizes, initialized to zero.
     * @param arima_size Number of ARIMA parameters
     * @param garch_size Number of GARCH parameters
     */
    ModelParameters(std::size_t arima_size, std::size_t garch_size)
        : arima_params_(arima_size, 0.0), garch_params_(garch_size, 0.0) {}

    /**
     * @brief Get const reference to ARIMA parameters.
     * @return Const reference to ARIMA ParameterVector
     */
    [[nodiscard]] const ParameterVector& arimaParams() const noexcept { return arima_params_; }

    /**
     * @brief Get mutable reference to ARIMA parameters.
     * @return Reference to ARIMA ParameterVector
     */
    ParameterVector& arimaParams() noexcept { return arima_params_; }

    /**
     * @brief Get const reference to GARCH parameters.
     * @return Const reference to GARCH ParameterVector
     */
    [[nodiscard]] const ParameterVector& garchParams() const noexcept { return garch_params_; }

    /**
     * @brief Get mutable reference to GARCH parameters.
     * @return Reference to GARCH ParameterVector
     */
    ParameterVector& garchParams() noexcept { return garch_params_; }

    /**
     * @brief Get total number of parameters (ARIMA + GARCH).
     * @return Total parameter count
     */
    [[nodiscard]] std::size_t totalSize() const noexcept {
        return arima_params_.size() + garch_params_.size();
    }

    /**
     * @brief Check if parameters are empty (both ARIMA and GARCH).
     * @return true if both parameter vectors are empty, false otherwise
     */
    [[nodiscard]] bool empty() const noexcept {
        return arima_params_.empty() && garch_params_.empty();
    }

    /**
     * @brief Get number of ARIMA parameters.
     * @return Size of ARIMA parameter vector
     */
    [[nodiscard]] std::size_t arimaSize() const noexcept { return arima_params_.size(); }

    /**
     * @brief Get number of GARCH parameters.
     * @return Size of GARCH parameter vector
     */
    [[nodiscard]] std::size_t garchSize() const noexcept { return garch_params_.size(); }

private:
    ParameterVector arima_params_;  // Serialization-friendly: no raw pointers
    ParameterVector garch_params_;  // Serialization-friendly: no raw pointers
};

}  // namespace ag::estimation
