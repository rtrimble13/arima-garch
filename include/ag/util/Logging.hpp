#pragma once

#include <string>
#include <string_view>

#include <fmt/core.h>
#include <fmt/format.h>

namespace ag {

// Log levels
enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error
};

// Logger class for formatted logging
class Logger {
public:
    // Get singleton instance
    static Logger& instance();

    // Set minimum log level
    void setLevel(LogLevel level);

    // Get current log level
    LogLevel level() const;

    // Log methods
    template <typename... Args>
    void debug(fmt::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Debug, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(fmt::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Info, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warning(fmt::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Warning, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(fmt::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Error, fmt, std::forward<Args>(args)...);
    }

private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    template <typename... Args>
    void log(LogLevel level, fmt::format_string<Args...> fmt, Args&&... args) {
        if (level >= current_level_) {
            logImpl(level, fmt::format(fmt, std::forward<Args>(args)...));
        }
    }

    void logImpl(LogLevel level, const std::string& message);

    LogLevel current_level_ = LogLevel::Info;
};

// Convenience functions for global logging
template <typename... Args>
inline void log_debug(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::instance().debug(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void log_info(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::instance().info(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void log_warning(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::instance().warning(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void log_error(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::instance().error(fmt, std::forward<Args>(args)...);
}

}  // namespace ag
