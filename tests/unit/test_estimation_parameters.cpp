#include "ag/estimation/Constraints.hpp"

#include <stdexcept>
#include <vector>

#include "test_framework.hpp"

using ag::estimation::ModelParameters;
using ag::estimation::ParameterVector;

// ============================================================================
// ParameterVector Tests
// ============================================================================

// Test default constructor
TEST(parameter_vector_default_constructor) {
    ParameterVector vec;
    REQUIRE(vec.size() == 0);
    REQUIRE(vec.empty());
}

// Test size constructor with default value
TEST(parameter_vector_size_constructor_default) {
    ParameterVector vec(5);
    REQUIRE(vec.size() == 5);
    REQUIRE(!vec.empty());
    for (std::size_t i = 0; i < 5; ++i) {
        REQUIRE_APPROX(vec[i], 0.0, 1e-10);
    }
}

// Test size constructor with custom initial value
TEST(parameter_vector_size_constructor_custom) {
    ParameterVector vec(3, 1.5);
    REQUIRE(vec.size() == 3);
    REQUIRE(!vec.empty());
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_APPROX(vec[i], 1.5, 1e-10);
    }
}

// Test construction from std::vector
TEST(parameter_vector_from_vector) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    ParameterVector vec(values);
    REQUIRE(vec.size() == 4);
    REQUIRE_APPROX(vec[0], 1.0, 1e-10);
    REQUIRE_APPROX(vec[1], 2.0, 1e-10);
    REQUIRE_APPROX(vec[2], 3.0, 1e-10);
    REQUIRE_APPROX(vec[3], 4.0, 1e-10);
}

// Test construction from std::vector with move semantics
TEST(parameter_vector_from_vector_move) {
    std::vector<double> values = {5.0, 6.0, 7.0};
    ParameterVector vec(std::move(values));
    REQUIRE(vec.size() == 3);
    REQUIRE_APPROX(vec[0], 5.0, 1e-10);
    REQUIRE_APPROX(vec[1], 6.0, 1e-10);
    REQUIRE_APPROX(vec[2], 7.0, 1e-10);
}

// Test element access and modification
TEST(parameter_vector_element_access) {
    ParameterVector vec(3, 0.0);
    vec[0] = 10.0;
    vec[1] = 20.0;
    vec[2] = 30.0;

    REQUIRE_APPROX(vec[0], 10.0, 1e-10);
    REQUIRE_APPROX(vec[1], 20.0, 1e-10);
    REQUIRE_APPROX(vec[2], 30.0, 1e-10);
}

// Test const element access
TEST(parameter_vector_const_access) {
    const ParameterVector vec(std::vector<double>{1.0, 2.0, 3.0});
    REQUIRE_APPROX(vec[0], 1.0, 1e-10);
    REQUIRE_APPROX(vec[1], 2.0, 1e-10);
    REQUIRE_APPROX(vec[2], 3.0, 1e-10);
}

// Test out of bounds access throws
TEST(parameter_vector_out_of_bounds) {
    ParameterVector vec(3, 0.0);
    bool caught_exception = false;
    try {
        double val = vec[5];
        (void)val;  // Suppress unused variable warning
    } catch (const std::out_of_range& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// Test values() accessor
TEST(parameter_vector_values_accessor) {
    ParameterVector vec(std::vector<double>{1.0, 2.0, 3.0});
    const std::vector<double>& values = vec.values();
    REQUIRE(values.size() == 3);
    REQUIRE_APPROX(values[0], 1.0, 1e-10);
    REQUIRE_APPROX(values[1], 2.0, 1e-10);
    REQUIRE_APPROX(values[2], 3.0, 1e-10);
}

// Test mutable values() accessor
TEST(parameter_vector_values_mutable) {
    ParameterVector vec(3, 0.0);
    std::vector<double>& values = vec.values();
    values[0] = 5.0;
    values[1] = 6.0;
    values[2] = 7.0;

    REQUIRE_APPROX(vec[0], 5.0, 1e-10);
    REQUIRE_APPROX(vec[1], 6.0, 1e-10);
    REQUIRE_APPROX(vec[2], 7.0, 1e-10);
}

// Test resize
TEST(parameter_vector_resize) {
    ParameterVector vec(3, 1.0);
    REQUIRE(vec.size() == 3);

    vec.resize(5, 2.0);
    REQUIRE(vec.size() == 5);
    REQUIRE_APPROX(vec[0], 1.0, 1e-10);  // Original values preserved
    REQUIRE_APPROX(vec[1], 1.0, 1e-10);
    REQUIRE_APPROX(vec[2], 1.0, 1e-10);
    REQUIRE_APPROX(vec[3], 2.0, 1e-10);  // New values
    REQUIRE_APPROX(vec[4], 2.0, 1e-10);

    vec.resize(2);
    REQUIRE(vec.size() == 2);
    REQUIRE_APPROX(vec[0], 1.0, 1e-10);
    REQUIRE_APPROX(vec[1], 1.0, 1e-10);
}

// Test clear
TEST(parameter_vector_clear) {
    ParameterVector vec(5, 1.0);
    REQUIRE(vec.size() == 5);
    REQUIRE(!vec.empty());

    vec.clear();
    REQUIRE(vec.size() == 0);
    REQUIRE(vec.empty());
}

// Test empty vector
TEST(parameter_vector_empty_vector) {
    ParameterVector vec(0);
    REQUIRE(vec.size() == 0);
    REQUIRE(vec.empty());
}

// ============================================================================
// ModelParameters Tests
// ============================================================================

// Test default constructor
TEST(model_parameters_default_constructor) {
    ModelParameters params;
    REQUIRE(params.arimaSize() == 0);
    REQUIRE(params.garchSize() == 0);
    REQUIRE(params.totalSize() == 0);
    REQUIRE(params.empty());
}

// Test construction with sizes
TEST(model_parameters_size_constructor) {
    ModelParameters params(3, 4);
    REQUIRE(params.arimaSize() == 3);
    REQUIRE(params.garchSize() == 4);
    REQUIRE(params.totalSize() == 7);
    REQUIRE(!params.empty());

    // Verify initialized to zero
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_APPROX(params.arimaParams()[i], 0.0, 1e-10);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_APPROX(params.garchParams()[i], 0.0, 1e-10);
    }
}

// Test construction with ParameterVector objects
TEST(model_parameters_vector_constructor) {
    ParameterVector arima_vec(std::vector<double>{1.0, 2.0, 3.0});
    ParameterVector garch_vec(std::vector<double>{4.0, 5.0});

    ModelParameters params(arima_vec, garch_vec);
    REQUIRE(params.arimaSize() == 3);
    REQUIRE(params.garchSize() == 2);
    REQUIRE(params.totalSize() == 5);
    REQUIRE(!params.empty());

    REQUIRE_APPROX(params.arimaParams()[0], 1.0, 1e-10);
    REQUIRE_APPROX(params.arimaParams()[1], 2.0, 1e-10);
    REQUIRE_APPROX(params.arimaParams()[2], 3.0, 1e-10);
    REQUIRE_APPROX(params.garchParams()[0], 4.0, 1e-10);
    REQUIRE_APPROX(params.garchParams()[1], 5.0, 1e-10);
}

// Test construction with move semantics
TEST(model_parameters_vector_constructor_move) {
    ParameterVector arima_vec(std::vector<double>{1.0, 2.0});
    ParameterVector garch_vec(std::vector<double>{3.0, 4.0, 5.0});

    ModelParameters params(std::move(arima_vec), std::move(garch_vec));
    REQUIRE(params.arimaSize() == 2);
    REQUIRE(params.garchSize() == 3);
    REQUIRE(params.totalSize() == 5);
}

// Test ARIMA parameter access
TEST(model_parameters_arima_access) {
    ModelParameters params(3, 2);
    params.arimaParams()[0] = 1.5;
    params.arimaParams()[1] = 2.5;
    params.arimaParams()[2] = 3.5;

    REQUIRE_APPROX(params.arimaParams()[0], 1.5, 1e-10);
    REQUIRE_APPROX(params.arimaParams()[1], 2.5, 1e-10);
    REQUIRE_APPROX(params.arimaParams()[2], 3.5, 1e-10);
}

// Test GARCH parameter access
TEST(model_parameters_garch_access) {
    ModelParameters params(2, 3);
    params.garchParams()[0] = 0.1;
    params.garchParams()[1] = 0.2;
    params.garchParams()[2] = 0.7;

    REQUIRE_APPROX(params.garchParams()[0], 0.1, 1e-10);
    REQUIRE_APPROX(params.garchParams()[1], 0.2, 1e-10);
    REQUIRE_APPROX(params.garchParams()[2], 0.7, 1e-10);
}

// Test const parameter access
TEST(model_parameters_const_access) {
    ParameterVector arima_vec(std::vector<double>{1.0, 2.0});
    ParameterVector garch_vec(std::vector<double>{3.0, 4.0});
    const ModelParameters params(arima_vec, garch_vec);

    REQUIRE(params.arimaSize() == 2);
    REQUIRE(params.garchSize() == 2);
    REQUIRE_APPROX(params.arimaParams()[0], 1.0, 1e-10);
    REQUIRE_APPROX(params.garchParams()[1], 4.0, 1e-10);
}

// Test zero ARIMA parameters
TEST(model_parameters_zero_arima) {
    ModelParameters params(0, 3);
    REQUIRE(params.arimaSize() == 0);
    REQUIRE(params.garchSize() == 3);
    REQUIRE(params.totalSize() == 3);
    REQUIRE(!params.empty());  // Not empty because GARCH has parameters
}

// Test zero GARCH parameters
TEST(model_parameters_zero_garch) {
    ModelParameters params(4, 0);
    REQUIRE(params.arimaSize() == 4);
    REQUIRE(params.garchSize() == 0);
    REQUIRE(params.totalSize() == 4);
    REQUIRE(!params.empty());  // Not empty because ARIMA has parameters
}

// Test both empty
TEST(model_parameters_both_empty) {
    ModelParameters params(0, 0);
    REQUIRE(params.arimaSize() == 0);
    REQUIRE(params.garchSize() == 0);
    REQUIRE(params.totalSize() == 0);
    REQUIRE(params.empty());
}

// Test parameter modification through accessors
TEST(model_parameters_modification) {
    ModelParameters params(2, 2);

    // Modify ARIMA parameters
    params.arimaParams()[0] = 10.0;
    params.arimaParams()[1] = 20.0;

    // Modify GARCH parameters
    params.garchParams()[0] = 30.0;
    params.garchParams()[1] = 40.0;

    // Verify modifications
    REQUIRE_APPROX(params.arimaParams()[0], 10.0, 1e-10);
    REQUIRE_APPROX(params.arimaParams()[1], 20.0, 1e-10);
    REQUIRE_APPROX(params.garchParams()[0], 30.0, 1e-10);
    REQUIRE_APPROX(params.garchParams()[1], 40.0, 1e-10);
}

// Test realistic ARIMA(1,1,1)-GARCH(1,1) parameter structure
TEST(model_parameters_arima_garch_11) {
    // ARIMA(1,1,1): intercept + 1 AR coef + 1 MA coef = 3 parameters
    // GARCH(1,1): omega + 1 ARCH coef + 1 GARCH coef = 3 parameters
    ModelParameters params(3, 3);

    // Set ARIMA parameters: [intercept, AR(1), MA(1)]
    params.arimaParams()[0] = 0.5;  // intercept
    params.arimaParams()[1] = 0.7;  // AR(1)
    params.arimaParams()[2] = 0.3;  // MA(1)

    // Set GARCH parameters: [omega, ARCH(1), GARCH(1)]
    params.garchParams()[0] = 0.01;  // omega
    params.garchParams()[1] = 0.1;   // ARCH(1)
    params.garchParams()[2] = 0.85;  // GARCH(1)

    REQUIRE(params.totalSize() == 6);
    REQUIRE_APPROX(params.arimaParams()[0], 0.5, 1e-10);
    REQUIRE_APPROX(params.garchParams()[2], 0.85, 1e-10);
}

int main() {
    report_test_results("Estimation Parameters");
    return get_test_result();
}
