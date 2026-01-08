#include "ag/cli/CliUtils.hpp"

#include "ag/io/CsvReader.hpp"

#include <sstream>
#include <stdexcept>

namespace ag::cli {

std::tuple<int, int, int> parseArimaOrder(const std::string& order) {
    std::istringstream iss(order);
    int p, d, q;
    char comma1, comma2;
    if (!(iss >> p >> comma1 >> d >> comma2 >> q) || comma1 != ',' || comma2 != ',') {
        throw std::invalid_argument("Invalid ARIMA order format. Use p,d,q (e.g., 1,1,1)");
    }
    return {p, d, q};
}

std::tuple<int, int> parseGarchOrder(const std::string& order) {
    std::istringstream iss(order);
    int p, q;
    char comma;
    if (!(iss >> p >> comma >> q) || comma != ',') {
        throw std::invalid_argument("Invalid GARCH order format. Use p,q (e.g., 1,1)");
    }
    return {p, q};
}

std::vector<double> loadData(const std::string& filepath, bool has_header) {
    ag::io::CsvReaderOptions options;
    options.has_header = has_header;
    // Use auto-detection for value column (default)
    // options.value_column is already set to std::numeric_limits<std::size_t>::max()

    auto result = ag::io::CsvReader::read(filepath, options);
    if (!result) {
        throw std::runtime_error("Failed to read data from file: " + filepath);
    }

    // Convert TimeSeries to vector
    std::vector<double> data;
    const auto& ts = *result;
    data.reserve(ts.size());
    for (std::size_t i = 0; i < ts.size(); ++i) {
        data.push_back(ts[i]);
    }
    return data;
}

}  // namespace ag::cli
