#pragma once

#include <chrono>
#include <string>

namespace ag {

// High-resolution timer for performance measurement
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    // Constructor - starts the timer
    Timer();

    // Start or restart the timer
    void start();

    // Stop the timer and return elapsed time in seconds
    double stop();

    // Get elapsed time without stopping the timer (in seconds)
    double elapsed() const;

    // Check if timer is running
    bool isRunning() const;

    // Get elapsed time in milliseconds
    double elapsedMs() const;

    // Get elapsed time in microseconds
    double elapsedUs() const;

    // Get elapsed time in nanoseconds
    long long elapsedNs() const;

private:
    TimePoint start_time_;
    TimePoint stop_time_;
    bool running_;
};

// Scoped timer that logs elapsed time on destruction
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

    // Disable copying
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string name_;
    Timer timer_;
};

}  // namespace ag
