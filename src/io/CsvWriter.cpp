#include "ag/io/CsvWriter.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace ag::io {

expected<std::string, CsvWriteError> CsvWriter::write_to_string(
    const data::TimeSeries& timeseries, const CsvWriterOptions& options) {
    // Check for index column size mismatch
    if (!options.index_column.empty() && options.index_column.size() != timeseries.size()) {
        return unexpected(CsvWriteError{"Index column size (" +
                                        std::to_string(options.index_column.size()) +
                                        ") does not match time series size (" +
                                        std::to_string(timeseries.size()) + ")"});
    }

    std::ostringstream oss;

    // Set precision
    if (options.precision >= 0) {
        oss << std::fixed << std::setprecision(options.precision);
    } else {
        // Maximum precision
        oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    }

    // Write header if configured
    bool has_index = !options.index_column.empty();
    bool write_header = !options.value_header.empty() ||
                        (has_index && !options.index_header.empty());
    if (write_header) {
        if (has_index && !options.index_header.empty()) {
            oss << options.index_header << options.delimiter;
        }
        if (!options.value_header.empty()) {
            oss << options.value_header;
        }
        oss << '\n';
    }

    // Write data rows
    for (std::size_t i = 0; i < timeseries.size(); ++i) {
        // Write index column if provided
        if (!options.index_column.empty()) {
            oss << options.index_column[i] << options.delimiter;
        }

        // Write value
        oss << timeseries[i] << '\n';
    }

    return oss.str();
}

expected<Success, CsvWriteError> CsvWriter::write(const std::filesystem::path& filepath,
                                                   const data::TimeSeries& timeseries,
                                                   const CsvWriterOptions& options) {
    // Generate CSV content
    auto content_result = write_to_string(timeseries, options);
    if (!content_result) {
        return unexpected(content_result.error());
    }

    // Open file for writing
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return unexpected(CsvWriteError{"Failed to open file for writing: " + filepath.string()});
    }

    // Write content
    file << *content_result;
    file.close();

    if (file.fail()) {
        return unexpected(CsvWriteError{"Failed to write to file: " + filepath.string()});
    }

    return Success{};
}

}  // namespace ag::io
