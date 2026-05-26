#pragma once

// Subcommand handler declarations for the `ag` CLI. Each function returns
// a process exit code (0 = success, non-zero = error).

#include <exception>
#include <string>

#include <fmt/core.h>

namespace ag::cli {

/**
 * @brief Run a callable, mapping any std::exception it throws to exit code 1
 *        with the exception message printed to stdout. Shared by every CLI
 *        handler so error formatting stays uniform.
 */
template <typename Func>
int executeWithErrorHandling(Func&& func) {
    try {
        return func();
    } catch (const std::exception& e) {
        fmt::print("Error: {}\n", e.what());
        return 1;
    }
}

int handleFit(const std::string& dataFile, const std::string& arimaOrder,
              const std::string& garchOrder, const std::string& outputFile, bool no_header,
              bool use_student_t, double student_t_df);

int handleSelect(const std::string& dataFile, int maxP, int maxD, int maxQ, int maxGarchP,
                 int maxGarchQ, const std::string& criterion, const std::string& outputFile,
                 int topK, bool no_header);

int handleForecast(const std::string& modelFile, int horizon, const std::string& outputFile);

int handleSimulate(const std::string& arimaOrder, const std::string& garchOrder, int length,
                   unsigned int seed, const std::string& outputFile, bool use_student_t,
                   double student_t_df);

int handleSimulateFromModel(const std::string& modelFile, int numPaths, int length,
                            unsigned int seed, const std::string& outputFile, bool computeStats);

int handleDiagnostics(const std::string& modelFile, const std::string& dataFile,
                      const std::string& outputFile, bool no_header);

}  // namespace ag::cli
