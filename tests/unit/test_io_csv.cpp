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

// Test trimming leading empty values
TEST(csv_read_trim_leading_empty) {
    std::string csv_content = "\n\n1.5\n2.3\n1.8\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test trimming trailing empty values
TEST(csv_read_trim_trailing_empty) {
    std::string csv_content = "1.5\n2.3\n1.8\n\n\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test trimming leading NA values
TEST(csv_read_trim_leading_na) {
    std::string csv_content = "NA\nNA\n1.5\n2.3\n1.8\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test trimming trailing NULL values
TEST(csv_read_trim_trailing_null) {
    std::string csv_content = "1.5\n2.3\n1.8\nNULL\nNULL\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test trimming both leading and trailing empty values
TEST(csv_read_trim_both_ends) {
    std::string csv_content = "NA\n\n1.5\n2.3\n1.8\n\nNULL\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 3);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
    REQUIRE_APPROX(ts[2], 1.8, 1e-10);
}

// Test with header and leading/trailing empty values
TEST(csv_read_with_header_trim_empty) {
    std::string csv_content = "Value\nNA\n1.5\n2.3\n1.8\nNULL\n";

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

// Test multiple columns with leading/trailing empty in value column
TEST(csv_read_multiple_columns_trim_empty) {
    std::string csv_content = "Date,Value\n"
                              "2020-01-01,NA\n"
                              "2020-01-02,1.5\n"
                              "2020-01-03,2.3\n"
                              "2020-01-04,1.8\n"
                              "2020-01-05,NULL\n";

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

// Test case-insensitive null values
TEST(csv_read_case_insensitive_nulls) {
    std::string csv_content = "na\nNa\n1.5\n2.3\nnull\nNULL\nNaN\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 2);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
}

// Test error on empty value in middle of data
TEST(csv_read_error_empty_in_middle) {
    std::string csv_content = "1.5\nNA\n2.3\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(!result.has_value());
}

// Test error when all values are empty/null
TEST(csv_read_error_all_empty) {
    std::string csv_content = "NA\nNULL\n\nNaN\n";

    auto result = CsvReader::read_from_string(csv_content);
    REQUIRE(!result.has_value());
}

// Test auto-detection with multiple columns and empty values
TEST(csv_read_auto_detect_with_empty) {
    std::string csv_content = "Date,Value\n"
                              "2020-01-01,NA\n"
                              "2020-01-02,1.5\n"
                              "2020-01-03,2.3\n"
                              "2020-01-04,NULL\n";

    CsvReaderOptions options;
    options.has_header = true;
    // Let it auto-detect the Value column (should skip Date column)

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 2);
    REQUIRE_APPROX(ts[0], 1.5, 1e-10);
    REQUIRE_APPROX(ts[1], 2.3, 1e-10);
}

// Test files without header show column labels as column1, column2, etc.
TEST(csv_read_no_header_error_message) {
    std::string csv_content = "1.5,2.3\n3.2,abc\n";

    CsvReaderOptions options;
    options.has_header = false;
    options.value_column = 1;

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(!result.has_value());
    // Error message should contain "column2" since it's 0-indexed column 1
}

// Test reading CSV with empty initial value (trailing comma)
TEST(csv_read_empty_initial_value) {
    std::string csv_content = "date,value\n2025-01-01,\n2025-01-02,0.1\n2025-01-03,-0.1\n";

    CsvReaderOptions options;
    options.has_header = true;
    options.value_column = 1;

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 2);
    REQUIRE_APPROX(ts[0], 0.1, 1e-10);
    REQUIRE_APPROX(ts[1], -0.1, 1e-10);
}

// Test reading CSV with multiple trailing empty values
TEST(csv_read_multiple_trailing_empty) {
    std::string csv_content = "date,value\n2025-01-01,\n2025-01-02,\n2025-01-03,0.1\n2025-01-04,-0.1\n";

    CsvReaderOptions options;
    options.has_header = true;
    options.value_column = 1;

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 2);
    REQUIRE_APPROX(ts[0], 0.1, 1e-10);
    REQUIRE_APPROX(ts[1], -0.1, 1e-10);
}

// Test reading CSV with trailing empty at end
TEST(csv_read_trailing_empty_at_end) {
    std::string csv_content = "date,value\n2025-01-01,0.1\n2025-01-02,-0.1\n2025-01-03,\n";

    CsvReaderOptions options;
    options.has_header = true;
    options.value_column = 1;

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(result.has_value());

    const auto& ts = *result;
    REQUIRE(ts.size() == 2);
    REQUIRE_APPROX(ts[0], 0.1, 1e-10);
    REQUIRE_APPROX(ts[1], -0.1, 1e-10);
}

// Test reading CSV with empty fields interspersed (should fail)
TEST(csv_read_empty_in_middle_with_trailing) {
    std::string csv_content = "date,value\n2025-01-01,0.1\n2025-01-02,\n2025-01-03,-0.1\n";

    CsvReaderOptions options;
    options.has_header = true;
    options.value_column = 1;

    auto result = CsvReader::read_from_string(csv_content, options);
    REQUIRE(!result.has_value());
    // Should fail because there's an empty value in the middle
}

int main() {
    report_test_results("CSV IO Tests");
    return get_test_result();
}
