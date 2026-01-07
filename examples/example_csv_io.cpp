#include "ag/data/TimeSeries.hpp"
#include "ag/io/CsvReader.hpp"
#include "ag/io/CsvWriter.hpp"

#include <iostream>

int main() {
    // Example 1: Read from a simple CSV file
    std::cout << "=== Example 1: Reading simple CSV ===" << std::endl;
    auto result1 = ag::io::CsvReader::read("../../tests/fixtures/simple.csv");
    if (result1.has_value()) {
        const auto& ts = *result1;
        std::cout << "Read " << ts.size() << " values" << std::endl;
        std::cout << "Mean: " << ts.mean() << std::endl;
        std::cout << "First value: " << ts[0] << std::endl;
        std::cout << "Last value: " << ts[ts.size() - 1] << std::endl;
    } else {
        std::cout << "Error: " << result1.error().message << std::endl;
    }

    // Example 2: Read from a CSV with date column
    std::cout << "\n=== Example 2: Reading CSV with date column ===" << std::endl;
    ag::io::CsvReaderOptions options;
    options.has_header = true;
    options.value_column = 1;  // Values are in second column

    auto result2 = ag::io::CsvReader::read("../../tests/fixtures/with_date.csv", options);
    if (result2.has_value()) {
        const auto& ts = *result2;
        std::cout << "Read " << ts.size() << " values" << std::endl;
        std::cout << "Mean: " << ts.mean() << std::endl;
    } else {
        std::cout << "Error: " << result2.error().message << std::endl;
    }

    // Example 3: Auto-detect column and trim empty values
    std::cout << "\n=== Example 3: Auto-detect column with empty values ===" << std::endl;
    // Create a CSV string with leading/trailing empty values
    std::string csv_with_empty = "Date,Value\n"
                                 "2024-01-01,NA\n"
                                 "2024-01-02,10.5\n"
                                 "2024-01-03,11.2\n"
                                 "2024-01-04,12.8\n"
                                 "2024-01-05,NULL\n";

    ag::io::CsvReaderOptions auto_options;
    auto_options.has_header = true;
    // value_column is not set, so it will auto-detect the first numeric column

    auto result3 = ag::io::CsvReader::read_from_string(csv_with_empty, auto_options);
    if (result3.has_value()) {
        const auto& ts = *result3;
        std::cout << "Successfully auto-detected column and trimmed empty values" << std::endl;
        std::cout << "Read " << ts.size() << " values (after trimming)" << std::endl;
        std::cout << "Mean: " << ts.mean() << std::endl;
    } else {
        std::cout << "Error: " << result3.error().message << std::endl;
    }

    // Example 4: Create a time series and write to CSV
    std::cout << "\n=== Example 4: Writing CSV ===" << std::endl;
    ag::data::TimeSeries ts{1.1, 2.2, 3.3, 4.4, 5.5};

    ag::io::CsvWriterOptions write_options;
    write_options.value_header = "Value";

    auto write_result = ag::io::CsvWriter::write_to_string(ts, write_options);
    if (write_result.has_value()) {
        std::cout << "Generated CSV:\n" << *write_result << std::endl;
    } else {
        std::cout << "Error: " << write_result.error().message << std::endl;
    }

    // Example 5: Write CSV with index column
    std::cout << "=== Example 5: Writing CSV with date column ===" << std::endl;
    ag::data::TimeSeries ts2{10.5, 11.2, 12.8};

    ag::io::CsvWriterOptions write_options2;
    write_options2.index_column = {"2024-01-01", "2024-01-02", "2024-01-03"};
    write_options2.index_header = "Date";
    write_options2.value_header = "Price";

    auto write_result2 = ag::io::CsvWriter::write_to_string(ts2, write_options2);
    if (write_result2.has_value()) {
        std::cout << "Generated CSV:\n" << *write_result2 << std::endl;
    } else {
        std::cout << "Error: " << write_result2.error().message << std::endl;
    }

    return 0;
}
