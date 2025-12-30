#include "ag/io/CsvReader.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ag::io {

namespace {

// Helper function to trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    const auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    const auto end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

// Helper function to split a line by delimiter
std::vector<std::string> split_line(const std::string& line, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, delimiter)) {
        result.push_back(trim(item));
    }

    return result;
}

// Helper function to parse a string to double
expected<double, CsvReadError> parse_double(const std::string& str) {
    try {
        std::size_t pos;
        double value = std::stod(str, &pos);

        // Check if entire string was consumed
        if (pos != str.length()) {
            return unexpected(
                CsvReadError{"Invalid number format: '" + str + "' (extra characters)"});
        }

        return value;
    } catch (const std::invalid_argument&) {
        return unexpected(CsvReadError{"Invalid number format: '" + str + "'"});
    } catch (const std::out_of_range&) {
        return unexpected(CsvReadError{"Number out of range: '" + str + "'"});
    }
}

}  // namespace

expected<data::TimeSeries, CsvReadError> CsvReader::read(const std::filesystem::path& filepath,
                                                         const CsvReaderOptions& options) {
    // Check if file exists
    if (!std::filesystem::exists(filepath)) {
        return unexpected(CsvReadError{"File not found: " + filepath.string()});
    }

    // Open file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return unexpected(CsvReadError{"Failed to open file: " + filepath.string()});
    }

    // Read entire file content
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    // Parse content
    return read_from_string(buffer.str(), options);
}

expected<data::TimeSeries, CsvReadError>
CsvReader::read_from_string(std::string_view csv_content, const CsvReaderOptions& options) {
    std::vector<double> values;
    std::string content_str{csv_content};
    std::istringstream stream{content_str};
    std::string line;
    std::size_t line_number = 0;
    std::size_t detected_value_column = options.value_column;
    bool need_auto_detect = (options.value_column == std::numeric_limits<std::size_t>::max());

    while (std::getline(stream, line)) {
        line_number++;

        // Trim whitespace
        line = trim(line);

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        // Skip header if configured
        if (line_number == 1 && options.has_header) {
            continue;
        }

        // Split line by delimiter
        std::vector<std::string> columns = split_line(line, options.delimiter);

        if (columns.empty()) {
            continue;
        }

        // Auto-detect the first numeric column on first data line
        if (need_auto_detect) {
            detected_value_column = std::numeric_limits<std::size_t>::max();
            for (std::size_t i = 0; i < columns.size(); ++i) {
                auto test_result = parse_double(columns[i]);
                if (test_result) {
                    detected_value_column = i;
                    break;
                }
            }
            if (detected_value_column == std::numeric_limits<std::size_t>::max()) {
                return unexpected(CsvReadError{
                    "Could not auto-detect numeric column on line " + std::to_string(line_number) +
                    " - no columns contain valid numeric data"});
            }
            need_auto_detect = false;
        }

        // Check if value column index is valid
        if (detected_value_column >= columns.size()) {
            return unexpected(
                CsvReadError{"Value column index " + std::to_string(detected_value_column) +
                             " out of range on line " + std::to_string(line_number) + " (found " +
                             std::to_string(columns.size()) + " columns)"});
        }

        // Parse value
        auto value_result = parse_double(columns[detected_value_column]);
        if (!value_result) {
            return unexpected(CsvReadError{"Failed to parse value on line " +
                                           std::to_string(line_number) + ": " +
                                           value_result.error().message});
        }

        values.push_back(*value_result);
    }

    // Check if we read any values
    if (values.empty()) {
        return unexpected(CsvReadError{"No valid data found in CSV"});
    }

    return data::TimeSeries(std::move(values));
}

}  // namespace ag::io
