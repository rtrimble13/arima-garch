#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

// Simple test framework for utility tests
namespace ag_test {

extern int test_count;
extern int test_passed;

}  // namespace ag_test

// Helper functions
void report_test_results(const std::string& suite_name);
int get_test_result();

// Test macros
#define TEST(name)                                                                                 \
    void test_##name();                                                                            \
    struct test_##name##_runner {                                                                  \
        test_##name##_runner() {                                                                   \
            ag_test::test_count++;                                                                 \
            std::cout << "Running test: " << #name << "... ";                                      \
            try {                                                                                  \
                test_##name();                                                                     \
                ag_test::test_passed++;                                                            \
                std::cout << "PASSED\n";                                                           \
            } catch (const std::exception& e) {                                                    \
                std::cout << "FAILED: " << e.what() << "\n";                                       \
            } catch (...) {                                                                        \
                std::cout << "FAILED: unknown exception\n";                                        \
            }                                                                                      \
        }                                                                                          \
    } test_##name##_instance;                                                                      \
    void test_##name()

#define REQUIRE(expr)                                                                              \
    if (!(expr)) {                                                                                 \
        throw std::runtime_error("Assertion failed: " #expr);                                      \
    }

#define REQUIRE_APPROX(val, expected, tolerance)                                                   \
    if (std::abs((val) - (expected)) > (tolerance)) {                                              \
        throw std::runtime_error("Assertion failed: " #val                                         \
                                 " not approximately equal to " #expected);                        \
    }
