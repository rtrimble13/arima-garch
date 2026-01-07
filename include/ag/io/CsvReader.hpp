#pragma once

#include "ag/data/TimeSeries.hpp"
#include "ag/util/Expected.hpp"

#include <filesystem>
#include <limits>
#include <string>
#include <string_view>

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
     * @brief Column index containing the time series values (0-indexed).
     *
     * If set to std::numeric_limits<std::size_t>::max() (default), the reader
     * will automatically detect the first numeric column.
     * If the CSV has only one column, use 0.
     * If the CSV has a date/index column followed by values, use 1.
     *
     * Examples:
     * - For CSV with single column of values: value_column = 0
     * - For CSV with date,value columns: value_column = 1
     * - For automatic detection (searches for first numeric column): use default
     */
    std::size_t value_column = std::numeric_limits<std::size_t>::max();

    /**
     * @brief Whether the first row contains column headers.
     *
     * When true, the first row is treated as a header and used in error messages.
     * When false, columns are labeled as column1, column2, etc. in error messages.
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
 * - Multiple columns with automatic detection of first numeric column
 * - Optional date/index column followed by values
 * - Customizable delimiter and header options
 * - Automatic trimming of leading and trailing empty/null values
 *
 * Robust handling:
 * - Automatically detects first numeric column when value_column is not specified
 * - Trims leading empty/null values (empty strings, "NA", "NULL", "NaN", "none")
 * - Trims trailing empty/null values
 * - Provides helpful error messages with column names (from header) or column numbers
 * - Reports errors for empty/null values in the middle of data
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
 *
 * With leading/trailing empty values (automatically trimmed):
 * ```
 * Value
 * NA
 * 1.5
 * 2.3
 * 1.8
 * NULL
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
    static expected<data::TimeSeries, CsvReadError>
    read_from_string(std::string_view csv_content, const CsvReaderOptions& options = {});
};

}  // namespace ag::io
