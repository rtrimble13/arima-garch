#include "ag/io/CsvReader.hpp"
#include "ag/io/CsvWriter.hpp"

#include <filesystem>
#include <string>

#include "test_framework.hpp"

using ag::data::TimeSeries;
using ag::io::CsvReader;
using ag::io::CsvReaderOptions;
using ag::io::CsvWriter;
using ag::io::CsvWriterOptions;

// Test reading a simple CSV file without header
TEST(csv_read_simple) {
    // Path is relative to build/tests directory where ctest runs the executable
    std::filesystem::path fixture_path = "../../tests/fixtures/simple.csv";

    auto result = CsvReader::read(fixture_path);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 5);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
    REQUIRE_APPROX(ts[3], 3.2, 1e-10);
    REQUIRE_APPROX(ts[4], 2.9, 1e-10);
}

// Test reading a CSV file with header
TEST(csv_read_with_header) {
    std::filesystem::path fixture_path = "../../tests/fixtures/with_header.csv";

    CsvReaderOptions options;
    options.has_header = true;

    auto result = CsvReader::read(fixture_path, options);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 5);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
    REQUIRE_APPROX(ts[3], 3.2, 1e-10);
    REQUIRE_APPROX(ts[4], 2.9, 1e-10);
}

// Test reading a CSV file with date column
TEST(csv_read_with_date_column) {
    std::filesystem::path fixture_path = "../../tests/fixtures/with_date.csv";

    CsvReaderOptions options;
    options.has_header = true;
    options.value_column = 1;  // Values are in the second column

    auto result = CsvReader::read(fixture_path, options);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 5);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
    REQUIRE_APPROX(ts[3], 3.2, 1e-10);
    REQUIRE_APPROX(ts[4], 2.9, 1e-10);
}

// Test reading from string
TEST(csv_read_from_string) {
    std::string csv_content = "1.5\n2.3\n1.8\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test reading from string with header
TEST(csv_read_from_string_with_header) {
    std::string csv_content = "Value\n1.5\n2.3\n1.8\n";

    CsvReaderOptions options;
    options.has_header = true;

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test reading from string with multiple columns
TEST(csv_read_from_string_with_multiple_columns) {
    std::string csv_content = "Date,Value\n2020-01-01,1.5\n2020-01-02,2.3\n2020-01-03,1.8\n";

    CsvReaderOptions options;
    options.has_header = true;
    options.value_column = 1;

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test error handling: file not found
TEST(csv_read_error_file_not_found) {
    std::filesystem::path bad_path = "/nonexistent/path/file.csv";

    auto result = CsvReader::read(bad_path);
    REQUIRE(!result.has_value());
}

// Test error handling: invalid number format
TEST(csv_read_error_invalid_number) {
    std::string csv_content = "1.5\nabc\n3.2\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(!result.has_value());
}

// Test error handling: empty CSV
TEST(csv_read_error_empty_csv) {
    std::string csv_content = "";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(!result.has_value());
}

// Test error handling: column index out of range
TEST(csv_read_error_column_out_of_range) {
    std::string csv_content = "1.5,2.3\n3.2,4.1\n";

    CsvReaderOptions options;
    options.value_column = 5;  // Out of range

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(!result.has_value());
}

// Test writing to string
TEST(csv_write_to_string) {
    TimeSeries ts{1.5, 2.3, 1.8};

    auto result = CsvWriter::write_to_string(ts);
    REQUIRE(result.has_value());

    // Parse the result back and verify
    auto parse_result = CsvReader::read_from_string(*result);
    REQUIRE(parse_result.has_value());

    const auto& parsed_ts = *parse_result;
    REQUIRE(parsed_ts.size() == 3);
    REQUIRE_APPROX(parsed_ts[0], 1.5, 1e-5);
    REQUIRE_APPROX(parsed_ts[1], 2.3, 1e-5);
    REQUIRE_APPROX(parsed_ts[2], 1.8, 1e-5);
}

// Test writing to string with header
TEST(csv_write_to_string_with_header) {
    TimeSeries ts{1.5, 2.3, 1.8};

    CsvWriterOptions options;
    options.value_header = "Value";

    auto result = CsvWriter::write_to_string(ts, options);
    REQUIRE(result.has_value());

    // Parse the result back with header option
    CsvReaderOptions read_options;
    read_options.has_header = true;

    auto parse_result = CsvReader::read_from_string(*result, read_options);
    REQUIRE(parse_result.has_value());

    const auto& parsed_ts = *parse_result;
    REQUIRE(parsed_ts.size() == 3);
    REQUIRE_APPROX(parsed_ts[0], 1.5, 1e-5);
    REQUIRE_APPROX(parsed_ts[1], 2.3, 1e-5);
    REQUIRE_APPROX(parsed_ts[2], 1.8, 1e-5);
}

// Test writing to string with index column
TEST(csv_write_to_string_with_index) {
    TimeSeries ts{1.5, 2.3, 1.8};

    CsvWriterOptions options;
    options.index_column = {"2020-01-01", "2020-01-02", "2020-01-03"};
    options.index_header = "Date";
    options.value_header = "Value";

    auto result = CsvWriter::write_to_string(ts, options);
    REQUIRE(result.has_value());

    // Parse the result back
    CsvReaderOptions read_options;
    read_options.has_header = true;
    read_options.value_column = 1;

    auto parse_result = CsvReader::read_from_string(*result, read_options);
    REQUIRE(parse_result.has_value());

    const auto& parsed_ts = *parse_result;
    REQUIRE(parsed_ts.size() == 3);
    REQUIRE_APPROX(parsed_ts[0], 1.5, 1e-5);
    REQUIRE_APPROX(parsed_ts[1], 2.3, 1e-5);
    REQUIRE_APPROX(parsed_ts[2], 1.8, 1e-5);
}

// Test writing to file
TEST(csv_write_to_file) {
    TimeSeries ts{1.5, 2.3, 1.8, 3.2};

    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "test_output.csv";

    auto result = CsvWriter::write(temp_path, ts);
    REQUIRE(result.has_value());

    // Read back and verify
    auto read_result = CsvReader::read(temp_path);
    REQUIRE(read_result.has_value());

    const auto& parsed_ts = *read_result;
    REQUIRE(parsed_ts.size() == 4);
    REQUIRE_APPROX(parsed_ts[0], 1.5, 1e-5);
    REQUIRE_APPROX(parsed_ts[1], 2.3, 1e-5);
    REQUIRE_APPROX(parsed_ts[2], 1.8, 1e-5);
    REQUIRE_APPROX(parsed_ts[3], 3.2, 1e-5);

    // Clean up
    std::filesystem::remove(temp_path);
}

// Test error handling: index column size mismatch
TEST(csv_write_error_index_size_mismatch) {
    TimeSeries ts{1.5, 2.3, 1.8};

    CsvWriterOptions options;
    options.index_column = {"2020-01-01", "2020-01-02"};  // Size mismatch

    auto result = CsvWriter::write_to_string(ts, options);
    REQUIRE(!result.has_value());
}

// Test reading with whitespace handling
TEST(csv_read_whitespace_handling) {
    std::string csv_content = "  1.5  \n  2.3  \n  1.8  \n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test reading with empty lines
TEST(csv_read_skip_empty_lines) {
    std::string csv_content = "1.5\n\n2.3\n\n1.8\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test negative values
TEST(csv_read_negative_values) {
    std::string csv_content = "-1.5\n2.3\n-3.8\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], -1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], -3.8, 1e-10);
}

// Test scientific notation
TEST(csv_read_scientific_notation) {
    std::string csv_content = "1.5e2\n2.3e-1\n1.8e0\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 150.0, 1e-10);
    REQUIRE_APPROX(ts[1], 0.23, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

int main() {
    report_test_results("CSV IO Tests");
    return get_test_result();
}
