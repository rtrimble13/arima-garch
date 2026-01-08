/**
 * @file cli/CliUtils.hpp
 * @brief CLI utility functions for parsing command-line arguments and loading data.
 *
 * This module provides helper functions used by the CLI subcommand handlers.
 */

#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace ag::cli {

/**
 * @brief Parse ARIMA order string (e.g., "1,1,1" -> p=1, d=1, q=1)
 *
 * @param order String in format "p,d,q"
 * @return Tuple of (p, d, q) values
 * @throws std::invalid_argument if format is invalid
 */
std::tuple<int, int, int> parseArimaOrder(const std::string& order);

/**
 * @brief Parse GARCH order string (e.g., "1,1" -> p=1, q=1)
 *
 * @param order String in format "p,q"
 * @return Tuple of (p, q) values
 * @throws std::invalid_argument if format is invalid
 */
std::tuple<int, int> parseGarchOrder(const std::string& order);

/**
 * @brief Load time series data from CSV file
 *
 * Automatically detects the first numeric column in the CSV file.
 *
 * @param filepath Path to CSV file
 * @param has_header Whether the CSV file has a header row
 * @return Vector of time series values
 * @throws std::runtime_error if file cannot be read
 */
std::vector<double> loadData(const std::string& filepath, bool has_header = true);

}  // namespace ag::cli
