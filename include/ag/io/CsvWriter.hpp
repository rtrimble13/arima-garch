#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "ag/data/TimeSeries.hpp"
#include "ag/util/Expected.hpp"

namespace ag::io {

/**
 * @brief Error type for CSV writing operations.
 */
struct CsvWriteError {
    std::string message;
};

/**
 * @brief Empty type to represent successful operations that don't return a value.
 */
struct Success {};

/**
 * @brief Configuration options for CSV writing.
 */
struct CsvWriterOptions {
    /**
     * @brief Column header for the values column.
     *
     * If empty, no header row is written.
     */
    std::string value_header = "";

    /**
     * @brief Optional column headers for date/index column.
     *
     * If non-empty, an additional column will be written with these values.
     * The size must match the number of values in the time series.
     */
    std::vector<std::string> index_column = {};

    /**
     * @brief Column header for the index column (if index_column is provided).
     */
    std::string index_header = "";

    /**
     * @brief Delimiter character for separating columns.
     */
    char delimiter = ',';

    /**
     * @brief Number of decimal places for formatting values.
     *
     * Use -1 for maximum precision.
     */
    int precision = 6;
};

/**
 * @brief Writer for saving time series data to CSV files.
 *
 * Supports writing:
 * - Single column of values
 * - Optional date/index column followed by values
 * - Customizable delimiter, headers, and formatting options
 *
 * Example output formats:
 *
 * Single column (no header):
 * ```
 * 1.5
 * 2.3
 * 1.8
 * ```
 *
 * With header:
 * ```
 * Value
 * 1.5
 * 2.3
 * 1.8
 * ```
 *
 * With date column and headers:
 * ```
 * Date,Value
 * 2020-01-01,1.5
 * 2020-01-02,2.3
 * 2020-01-03,1.8
 * ```
 */
class CsvWriter {
public:
    /**
     * @brief Write a time series to a CSV file.
     *
     * @param filepath Path to the output CSV file
     * @param timeseries The time series data to write
     * @param options Writer configuration options
     * @return Expected containing Success on success, or CsvWriteError on failure
     */
    static expected<Success, CsvWriteError> write(const std::filesystem::path& filepath,
                                                   const data::TimeSeries& timeseries,
                                                   const CsvWriterOptions& options = {});

    /**
     * @brief Write a time series to a CSV string.
     *
     * @param timeseries The time series data to write
     * @param options Writer configuration options
     * @return Expected containing CSV string on success, or CsvWriteError on failure
     */
    static expected<std::string, CsvWriteError> write_to_string(
        const data::TimeSeries& timeseries, const CsvWriterOptions& options = {});
};

}  // namespace ag::io
