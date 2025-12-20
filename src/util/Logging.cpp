#include "ag/util/Logging.hpp"

#include <iostream>

#include <fmt/color.h>
#include <fmt/core.h>

namespace ag {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::setLevel(LogLevel level) {
    current_level_ = level;
}

LogLevel Logger::level() const {
    return current_level_;
}

void Logger::logImpl(LogLevel level, const std::string& message) {
    // Format with colors and level prefix
    switch (level) {
    case LogLevel::Debug:
        fmt::print(fg(fmt::color::gray), "[DEBUG] {}\n", message);
        break;
    case LogLevel::Info:
        fmt::print(fg(fmt::color::green), "[INFO] {}\n", message);
        break;
    case LogLevel::Warning:
        fmt::print(fg(fmt::color::yellow), "[WARNING] {}\n", message);
        break;
    case LogLevel::Error:
        fmt::print(stderr, fg(fmt::color::red), "[ERROR] {}\n", message);
        break;
    }
}

}  // namespace ag
