#include "ag/util/Logging.hpp"
#include "test_framework.hpp"
#include <iostream>
#include <sstream>

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
    std::cout << "\nNote: Log messages above are expected output from tests\n";
    ag_test::report_test_results("Logging Tests");
    return ag_test::get_test_result();
}
