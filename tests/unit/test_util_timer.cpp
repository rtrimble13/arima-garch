#include "ag/util/Timer.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

// Simple test framework
int test_count = 0;
int test_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct test_##name##_runner { \
        test_##name##_runner() { \
            test_count++; \
            std::cout << "Running test: " << #name << "... "; \
            try { \
                test_##name(); \
                test_passed++; \
                std::cout << "PASSED\n"; \
            } catch (const std::exception& e) { \
                std::cout << "FAILED: " << e.what() << "\n"; \
            } catch (...) { \
                std::cout << "FAILED: unknown exception\n"; \
            } \
        } \
    } test_##name##_instance; \
    void test_##name()

#define REQUIRE(expr) \
    if (!(expr)) { \
        throw std::runtime_error("Assertion failed: " #expr); \
    }

#define REQUIRE_APPROX(val, expected, tolerance) \
    if (std::abs((val) - (expected)) > (tolerance)) { \
        throw std::runtime_error("Assertion failed: " #val " not approximately equal to " #expected); \
    }

// Test basic timer functionality
TEST(timer_basic) {
    ag::Timer timer;
    REQUIRE(timer.isRunning());
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    double elapsed = timer.elapsed();
    REQUIRE(elapsed >= 0.095);  // Allow some tolerance
    REQUIRE(elapsed <= 0.150);  // Allow some tolerance
    REQUIRE(timer.isRunning());  // Still running after elapsed()
}

// Test timer start/stop
TEST(timer_start_stop) {
    ag::Timer timer;
    timer.start();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    double elapsed = timer.stop();
    REQUIRE(!timer.isRunning());
    REQUIRE(elapsed >= 0.045);
    REQUIRE(elapsed <= 0.100);
    
    // Elapsed should remain the same after stop
    double elapsed2 = timer.elapsed();
    REQUIRE_APPROX(elapsed, elapsed2, 0.001);
}

// Test timer restart
TEST(timer_restart) {
    ag::Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    timer.start();  // Restart
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    double elapsed = timer.elapsed();
    REQUIRE(elapsed >= 0.045);
    REQUIRE(elapsed <= 0.100);  // Should be ~50ms, not ~100ms
}

// Test elapsed time units
TEST(timer_elapsed_units) {
    ag::Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    double seconds = timer.elapsed();
    double ms = timer.elapsedMs();
    double us = timer.elapsedUs();
    long long ns = timer.elapsedNs();
    
    // Check rough conversions
    REQUIRE_APPROX(ms, seconds * 1000.0, 5.0);
    REQUIRE_APPROX(us, seconds * 1000000.0, 5000.0);
    REQUIRE_APPROX(static_cast<double>(ns), seconds * 1000000000.0, 5000000.0);
}

// Test timer precision
TEST(timer_precision) {
    ag::Timer timer;
    
    // Very short sleep
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    
    // Should be able to measure microseconds
    double us = timer.elapsedUs();
    REQUIRE(us > 0);
    REQUIRE(us < 1000);  // Should be less than 1ms
}

// Test multiple stop calls
TEST(timer_multiple_stops) {
    ag::Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    double elapsed1 = timer.stop();
    double elapsed2 = timer.stop();
    
    // Should return approximately the same value
    REQUIRE_APPROX(elapsed1, elapsed2, 0.001);
}

int main() {
    std::cout << "\n=== Timer Tests ===\n";
    std::cout << "\nResults: " << test_passed << "/" << test_count << " tests passed\n";
    return test_passed == test_count ? 0 : 1;
}
