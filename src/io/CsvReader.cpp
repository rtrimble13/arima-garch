#include "ag/io/CsvReader.hpp"

#include <algorithm>
#include <fstream>
#include <optional>
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

    // Handle trailing delimiter - if line ends with delimiter, add empty field
    if (!line.empty() && line.back() == delimiter) {
        result.push_back("");
    }

    return result;
}

// Helper function to check if a string represents an empty/null value
bool is_empty_or_null(const std::string& str) {
    if (str.empty()) {
        return true;
    }
    // Check for common null representations (case-insensitive)
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower == "na" || lower == "null" || lower == "nan" || lower == "none";
}

// Helper function to parse a string to double
// Returns empty optional for empty/null values, error for invalid numeric strings
expected<std::optional<double>, CsvReadError> parse_double_optional(const std::string& str) {
    // Check for empty/null values first
    if (is_empty_or_null(str)) {
        return std::optional<double>{};
    }

    try {
        std::size_t pos;
        double value = std::stod(str, &pos);

        // Check if entire string was consumed
        if (pos != str.length()) {
            return unexpected(
                CsvReadError{"Invalid number format: '" + str + "' (extra characters)"});
        }

        return std::optional<double>{value};
    } catch (const std::invalid_argument&) {
        return unexpected(CsvReadError{"Invalid number format: '" + str + "'"});
    } catch (const std::out_of_range&) {
        return unexpected(CsvReadError{"Number out of range: '" + str + "'"});
    }
}

// Helper function to generate column label for error messages
std::string get_column_label(std::size_t col_index, bool has_header,
                             const std::vector<std::string>& header_row) {
    if (has_header && !header_row.empty() && col_index < header_row.size()) {
        return "'" + header_row[col_index] + "'";
    }
    return "column" + std::to_string(col_index + 1);
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
    std::vector<std::string> header_row;
    std::vector<std::vector<std::string>> all_rows;
    std::string content_str{csv_content};
    std::istringstream stream{content_str};
    std::string line;
    std::size_t line_number = 0;

    // First pass: read all rows
    while (std::getline(stream, line)) {
        line_number++;

        // Trim whitespace
        line = trim(line);

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        // Split line by delimiter
        std::vector<std::string> columns = split_line(line, options.delimiter);

        if (columns.empty()) {
            continue;
        }

        // Store header if configured
        if (line_number == 1 && options.has_header) {
            header_row = columns;
            continue;
        }

        all_rows.push_back(columns);
    }

    // Check if we have any data rows
    if (all_rows.empty()) {
        return unexpected(CsvReadError{"No valid data found in CSV"});
    }

    // Determine which column to use
    std::size_t detected_value_column = options.value_column;
    bool need_auto_detect = (options.value_column == std::numeric_limits<std::size_t>::max());

    if (need_auto_detect) {
        // Auto-detect: find first column that has at least one valid numeric value
        detected_value_column = std::numeric_limits<std::size_t>::max();

        // Determine max columns across all rows
        std::size_t max_columns = 0;
        for (const auto& row : all_rows) {
            max_columns = std::max(max_columns, row.size());
        }

        // Try each column
        for (std::size_t col = 0; col < max_columns; ++col) {
            bool found_valid = false;
            for (const auto& row : all_rows) {
                if (col < row.size()) {
                    auto test_result = parse_double_optional(row[col]);
                    if (test_result && test_result->has_value()) {
                        found_valid = true;
                        break;
                    }
                }
            }
            if (found_valid) {
                detected_value_column = col;
                break;
            }
        }

        if (detected_value_column == std::numeric_limits<std::size_t>::max()) {
            return unexpected(
                CsvReadError{"Could not auto-detect numeric column - no valid numeric data found"});
        }
    }

    // Parse all values from the detected column
    std::vector<std::optional<double>> values;
    values.reserve(all_rows.size());

    for (std::size_t i = 0; i < all_rows.size(); ++i) {
        const auto& row = all_rows[i];

        // Check if value column index is valid for this row
        if (detected_value_column >= row.size()) {
            std::string col_label =
                get_column_label(detected_value_column, options.has_header, header_row);
            return unexpected(CsvReadError{
                "Value column " + col_label + " (index " + std::to_string(detected_value_column) +
                ") out of range on line " + std::to_string(i + 1 + (options.has_header ? 1 : 0)) +
                " (found " + std::to_string(row.size()) + " columns)"});
        }

        // Parse value (may be empty/null)
        auto value_result = parse_double_optional(row[detected_value_column]);
        if (!value_result) {
            std::string col_label =
                get_column_label(detected_value_column, options.has_header, header_row);
            return unexpected(CsvReadError{"Failed to parse value in " + col_label + " on line " +
                                           std::to_string(i + 1 + (options.has_header ? 1 : 0)) +
                                           ": " + value_result.error().message});
        }

        values.push_back(*value_result);
    }

    // Trim leading empty/null values
    std::size_t first_valid = 0;
    while (first_valid < values.size() && !values[first_valid].has_value()) {
        first_valid++;
    }

    // Trim trailing empty/null values
    std::size_t last_valid = values.size();
    while (last_valid > first_valid && !values[last_valid - 1].has_value()) {
        last_valid--;
    }

    // Check if there are any valid values after trimming
    if (first_valid >= last_valid) {
        return unexpected(
            CsvReadError{"No valid numeric data found in CSV after trimming empty values"});
    }

    // Extract the valid values (between first_valid and last_valid)
    std::vector<double> final_values;
    final_values.reserve(last_valid - first_valid);

    for (std::size_t i = first_valid; i < last_valid; ++i) {
        if (values[i].has_value()) {
            final_values.push_back(*values[i]);
        } else {
            // Found an empty/null value in the middle of the data
            std::string col_label =
                get_column_label(detected_value_column, options.has_header, header_row);
            return unexpected(CsvReadError{
                "Empty or null value found in " + col_label + " on line " +
                std::to_string(i + 1 + (options.has_header ? 1 : 0)) +
                ". Only leading and trailing empty values are automatically trimmed."});
        }
    }

    return data::TimeSeries(std::move(final_values));
}

}  // namespace ag::io
