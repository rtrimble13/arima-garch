#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/ArimaSpec.hpp"
#include "ag/models/GarchSpec.hpp"

#include <stdexcept>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::models::ArimaSpec;
using ag::models::GarchSpec;

// ============================================================================
// ArimaSpec Tests
// ============================================================================

// Test valid ARIMA(0,0,0) specification
TEST(arima_spec_zero_order) {
    ArimaSpec spec(0, 0, 0);
    REQUIRE(spec.p == 0);
    REQUIRE(spec.d == 0);
    REQUIRE(spec.q == 0);
    REQUIRE(spec.isZeroOrder());
    REQUIRE(!spec.hasDifferencing());
    REQUIRE(!spec.hasAR());
    REQUIRE(!spec.hasMA());
}

// Test valid ARIMA(1,0,0) specification
TEST(arima_spec_ar_only) {
    ArimaSpec spec(1, 0, 0);
    REQUIRE(spec.p == 1);
    REQUIRE(spec.d == 0);
    REQUIRE(spec.q == 0);
    REQUIRE(!spec.isZeroOrder());
    REQUIRE(!spec.hasDifferencing());
    REQUIRE(spec.hasAR());
    REQUIRE(!spec.hasMA());
}

// Test valid ARIMA(0,1,0) specification
TEST(arima_spec_differencing_only) {
    ArimaSpec spec(0, 1, 0);
    REQUIRE(spec.p == 0);
    REQUIRE(spec.d == 1);
    REQUIRE(spec.q == 0);
    REQUIRE(!spec.isZeroOrder());
    REQUIRE(spec.hasDifferencing());
    REQUIRE(!spec.hasAR());
    REQUIRE(!spec.hasMA());
}

// Test valid ARIMA(0,0,1) specification
TEST(arima_spec_ma_only) {
    ArimaSpec spec(0, 0, 1);
    REQUIRE(spec.p == 0);
    REQUIRE(spec.d == 0);
    REQUIRE(spec.q == 1);
    REQUIRE(!spec.isZeroOrder());
    REQUIRE(!spec.hasDifferencing());
    REQUIRE(!spec.hasAR());
    REQUIRE(spec.hasMA());
}

// Test valid ARIMA(1,1,1) specification
TEST(arima_spec_full_model) {
    ArimaSpec spec(1, 1, 1);
    REQUIRE(spec.p == 1);
    REQUIRE(spec.d == 1);
    REQUIRE(spec.q == 1);
    REQUIRE(!spec.isZeroOrder());
    REQUIRE(spec.hasDifferencing());
    REQUIRE(spec.hasAR());
    REQUIRE(spec.hasMA());
}

// Test valid ARIMA(5,2,3) specification
TEST(arima_spec_higher_order) {
    ArimaSpec spec(5, 2, 3);
    REQUIRE(spec.p == 5);
    REQUIRE(spec.d == 2);
    REQUIRE(spec.q == 3);
    REQUIRE(!spec.isZeroOrder());
    REQUIRE(spec.hasDifferencing());
    REQUIRE(spec.hasAR());
    REQUIRE(spec.hasMA());
}

// Test invalid ARIMA specification: negative p
TEST(arima_spec_negative_p) {
    bool caught_exception = false;
    try {
        ArimaSpec spec(-1, 0, 0);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        // Verify the error message contains relevant information
        std::string msg(e.what());
        REQUIRE(msg.find("p") != std::string::npos);
        REQUIRE(msg.find("non-negative") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid ARIMA specification: negative d
TEST(arima_spec_negative_d) {
    bool caught_exception = false;
    try {
        ArimaSpec spec(0, -1, 0);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("d") != std::string::npos);
        REQUIRE(msg.find("non-negative") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid ARIMA specification: negative q
TEST(arima_spec_negative_q) {
    bool caught_exception = false;
    try {
        ArimaSpec spec(0, 0, -1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("q") != std::string::npos);
        REQUIRE(msg.find("non-negative") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid ARIMA specification: all negative
TEST(arima_spec_all_negative) {
    bool caught_exception = false;
    try {
        ArimaSpec spec(-1, -1, -1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// ============================================================================
// GarchSpec Tests
// ============================================================================

// Test valid GARCH(1,1) specification (most common)
TEST(garch_spec_11) {
    GarchSpec spec(1, 1);
    REQUIRE(spec.p == 1);
    REQUIRE(spec.q == 1);
    REQUIRE(spec.isGarch11());
}

// Test valid GARCH(1,2) specification
TEST(garch_spec_12) {
    GarchSpec spec(1, 2);
    REQUIRE(spec.p == 1);
    REQUIRE(spec.q == 2);
    REQUIRE(!spec.isGarch11());
}

// Test valid GARCH(2,1) specification
TEST(garch_spec_21) {
    GarchSpec spec(2, 1);
    REQUIRE(spec.p == 2);
    REQUIRE(spec.q == 1);
    REQUIRE(!spec.isGarch11());
}

// Test valid GARCH(3,3) specification
TEST(garch_spec_higher_order) {
    GarchSpec spec(3, 3);
    REQUIRE(spec.p == 3);
    REQUIRE(spec.q == 3);
    REQUIRE(!spec.isGarch11());
}

// Test invalid GARCH specification: p = 0 but q != 0
TEST(garch_spec_zero_p) {
    bool caught_exception = false;
    try {
        GarchSpec spec(0, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("both be 0") != std::string::npos ||
                msg.find("both be >= 1") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid GARCH specification: q = 0 but p != 0
TEST(garch_spec_zero_q) {
    bool caught_exception = false;
    try {
        GarchSpec spec(1, 0);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("both be 0") != std::string::npos ||
                msg.find("both be >= 1") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test valid GARCH specification: both zero (ARIMA-only model)
TEST(garch_spec_both_zero) {
    GarchSpec spec(0, 0);
    REQUIRE(spec.p == 0);
    REQUIRE(spec.q == 0);
    REQUIRE(spec.isNull());
    REQUIRE(!spec.isGarch11());
}

// Test invalid GARCH specification: negative p
TEST(garch_spec_negative_p) {
    bool caught_exception = false;
    try {
        GarchSpec spec(-1, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("p") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid GARCH specification: negative q
TEST(garch_spec_negative_q) {
    bool caught_exception = false;
    try {
        GarchSpec spec(1, -1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("q") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid GARCH specification: both negative
TEST(garch_spec_both_negative) {
    bool caught_exception = false;
    try {
        GarchSpec spec(-1, -1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// ============================================================================
// ArimaGarchSpec Tests
// ============================================================================

// Test ARIMA-GARCH construction from specs
TEST(arima_garch_spec_from_specs) {
    ArimaSpec arima(1, 1, 1);
    GarchSpec garch(1, 1);
    ArimaGarchSpec spec(arima, garch);

    REQUIRE(spec.arimaSpec.p == 1);
    REQUIRE(spec.arimaSpec.d == 1);
    REQUIRE(spec.arimaSpec.q == 1);
    REQUIRE(spec.garchSpec.p == 1);
    REQUIRE(spec.garchSpec.q == 1);
}

// Test ARIMA-GARCH construction from parameters
TEST(arima_garch_spec_from_params) {
    ArimaGarchSpec spec(2, 1, 2, 1, 1);

    REQUIRE(spec.arimaSpec.p == 2);
    REQUIRE(spec.arimaSpec.d == 1);
    REQUIRE(spec.arimaSpec.q == 2);
    REQUIRE(spec.garchSpec.p == 1);
    REQUIRE(spec.garchSpec.q == 1);
}

// Test ARIMA-GARCH parameter counting
TEST(arima_garch_spec_param_count) {
    ArimaGarchSpec spec(1, 1, 1, 1, 1);

    // ARIMA params: p + q = 1 + 1 = 2
    REQUIRE(spec.arimaParamCount() == 2);

    // GARCH params: p + q = 1 + 1 = 2
    REQUIRE(spec.garchParamCount() == 2);

    // Total: ARIMA (p + q + intercept) + GARCH (p + q + omega)
    // = (1 + 1 + 1) + (1 + 1 + 1) = 6
    REQUIRE(spec.totalParamCount() == 6);
}

// Test ARIMA-GARCH with zero-order ARIMA
TEST(arima_garch_spec_zero_arima) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);

    REQUIRE(spec.arimaSpec.isZeroOrder());
    REQUIRE(spec.arimaParamCount() == 0);
    REQUIRE(spec.garchParamCount() == 2);

    // For zero-order ARIMA: no ARIMA params
    // GARCH: p + q + omega = 1 + 1 + 1 = 3
    REQUIRE(spec.totalParamCount() == 3);
}

// Test ARIMA-GARCH with higher-order models
TEST(arima_garch_spec_higher_order) {
    ArimaGarchSpec spec(3, 2, 2, 2, 2);

    REQUIRE(spec.arimaParamCount() == 5);  // 3 + 2
    REQUIRE(spec.garchParamCount() == 4);  // 2 + 2

    // ARIMA: (3 + 2 + 1) + GARCH: (2 + 2 + 1) = 11
    REQUIRE(spec.totalParamCount() == 11);
}

// Test invalid ARIMA-GARCH: invalid ARIMA
TEST(arima_garch_spec_invalid_arima) {
    bool caught_exception = false;
    try {
        ArimaGarchSpec spec(-1, 0, 0, 1, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// Test invalid ARIMA-GARCH: invalid GARCH
TEST(arima_garch_spec_invalid_garch) {
    bool caught_exception = false;
    try {
        ArimaGarchSpec spec(1, 1, 1, 0, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// Test invalid ARIMA-GARCH: both invalid
TEST(arima_garch_spec_both_invalid) {
    bool caught_exception = false;
    try {
        ArimaGarchSpec spec(-1, -1, -1, 0, 0);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

int main() {
    report_test_results("Model Specs");
    return get_test_result();
}
