#include "ag/util/Timer.hpp"

#include "ag/util/Logging.hpp"

#include <fmt/core.h>

namespace ag {

Timer::Timer() : start_time_(Clock::now()), stop_time_(Clock::now()), running_(true) {}

void Timer::start() {
    start_time_ = Clock::now();
    running_ = true;
}

double Timer::stop() {
    stop_time_ = Clock::now();
    running_ = false;
    Duration elapsed = std::chrono::duration_cast<Duration>(stop_time_ - start_time_);
    return elapsed.count();
}

double Timer::elapsed() const {
    TimePoint end_time = running_ ? Clock::now() : stop_time_;
    Duration elapsed = std::chrono::duration_cast<Duration>(end_time - start_time_);
    return elapsed.count();
}

bool Timer::isRunning() const {
    return running_;
}

double Timer::elapsedMs() const {
    return elapsed() * 1000.0;
}

double Timer::elapsedUs() const {
    return elapsed() * 1000000.0;
}

long long Timer::elapsedNs() const {
    return static_cast<long long>(elapsed() * 1000000000.0);
}

ScopedTimer::ScopedTimer(const std::string& name) : name_(name) {
    log_debug("Timer '{}' started", name_);
}

ScopedTimer::~ScopedTimer() {
    double elapsed = timer_.elapsed();
    log_info("Timer '{}' elapsed: {:.6f} seconds", name_, elapsed);
}

}  // namespace ag
