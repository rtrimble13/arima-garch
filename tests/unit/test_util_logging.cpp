#include "ag/util/Logging.hpp"
#include <iostream>
#include <sstream>

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

// Test logger singleton
TEST(logger_singleton) {
    ag::Logger& logger1 = ag::Logger::instance();
    ag::Logger& logger2 = ag::Logger::instance();
    
    REQUIRE(&logger1 == &logger2);
}

// Test log level setting
TEST(logger_level) {
    ag::Logger& logger = ag::Logger::instance();
    
    logger.setLevel(ag::LogLevel::Debug);
    REQUIRE(logger.level() == ag::LogLevel::Debug);
    
    logger.setLevel(ag::LogLevel::Info);
    REQUIRE(logger.level() == ag::LogLevel::Info);
    
    logger.setLevel(ag::LogLevel::Warning);
    REQUIRE(logger.level() == ag::LogLevel::Warning);
    
    logger.setLevel(ag::LogLevel::Error);
    REQUIRE(logger.level() == ag::LogLevel::Error);
}

// Test logging methods (just ensure they don't crash)
TEST(logger_methods) {
    ag::Logger& logger = ag::Logger::instance();
    logger.setLevel(ag::LogLevel::Debug);
    
    logger.debug("Debug message: {}", 42);
    logger.info("Info message: {}", "test");
    logger.warning("Warning message: {:.2f}", 3.14);
    logger.error("Error message: {} {}", "multi", "args");
}

// Test global logging functions
TEST(global_logging_functions) {
    ag::Logger::instance().setLevel(ag::LogLevel::Debug);
    
    ag::log_debug("Global debug: {}", 1);
    ag::log_info("Global info: {}", 2);
    ag::log_warning("Global warning: {}", 3);
    ag::log_error("Global error: {}", 4);
}

// Test log level filtering
TEST(logger_filtering) {
    ag::Logger& logger = ag::Logger::instance();
    
    // Set to Warning level - should only see Warning and Error
    logger.setLevel(ag::LogLevel::Warning);
    
    std::cout << "\n  Testing filtering (should see warning and error only):\n";
    logger.debug("  This debug should NOT appear");
    logger.info("  This info should NOT appear");
    logger.warning("  This warning SHOULD appear");
    logger.error("  This error SHOULD appear");
    std::cout << "  Filtering test complete\n";
    
    // Reset to Info for other tests
    logger.setLevel(ag::LogLevel::Info);
}

// Test formatting capabilities
TEST(logger_formatting) {
    ag::Logger& logger = ag::Logger::instance();
    logger.setLevel(ag::LogLevel::Info);
    
    logger.info("Integer: {}", 42);
    logger.info("Float: {:.3f}", 3.14159);
    logger.info("String: {}", "hello");
    logger.info("Multiple: {} {} {}", 1, 2.5, "three");
}

int main() {
    std::cout << "\n=== Logging Tests ===\n";
    std::cout << "\nNote: Log messages above are expected output from tests\n";
    std::cout << "\nResults: " << test_passed << "/" << test_count << " tests passed\n";
    return test_passed == test_count ? 0 : 1;
}
