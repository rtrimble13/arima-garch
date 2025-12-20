#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "ag/data/TimeSeries.hpp"
#include "ag/util/Expected.hpp"

namespace ag::io {

/**
 * @brief Error type for CSV reading operations.
 */
struct CsvReadError {
    std::string message;
};

/**
 * @brief Configuration options for CSV reading.
 */
struct CsvReaderOptions {
    /**
     * @brief Column index containing the time series values.
     *
     * If the CSV has only one column, use 0.
     * If the CSV has a date/index column followed by values, use 1.
     */
    std::size_t value_column = 0;

    /**
     * @brief Whether the first row contains column headers.
     */
    bool has_header = false;

    /**
     * @brief Delimiter character for separating columns.
     */
    char delimiter = ',';
};

/**
 * @brief Reader for loading time series data from CSV files.
 *
 * Supports common CSV formats:
 * - Single column of values (returns only)
 * - Optional date/index column followed by values
 * - Customizable delimiter and header options
 *
 * Example CSV formats:
 *
 * Single column (no header):
 * ```
 * 1.5
 * 2.3
 * 1.8
 * ```
 *
 * With date column and header:
 * ```
 * Date,Value
 * 2020-01-01,1.5
 * 2020-01-02,2.3
 * 2020-01-03,1.8
 * ```
 */
class CsvReader {
public:
    /**
     * @brief Read a time series from a CSV file.
     *
     * @param filepath Path to the CSV file
     * @param options Reader configuration options
     * @return Expected containing TimeSeries on success, or CsvReadError on failure
     */
    static expected<data::TimeSeries, CsvReadError> read(const std::filesystem::path& filepath,
                                                          const CsvReaderOptions& options = {});

    /**
     * @brief Read a time series from a CSV string.
     *
     * @param csv_content CSV content as a string
     * @param options Reader configuration options
     * @return Expected containing TimeSeries on success, or CsvReadError on failure
     */
    static expected<data::TimeSeries, CsvReadError> read_from_string(
        std::string_view csv_content, const CsvReaderOptions& options = {});
};

}  // namespace ag::io
