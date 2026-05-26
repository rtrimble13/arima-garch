#include "ag/util/Differencing.hpp"

#include <utility>

namespace ag::util {

std::vector<double> differenceSeries(const double* data, std::size_t size, int d) {
    if (size == 0 || data == nullptr) {
        return {};
    }

    std::vector<double> result(data, data + size);

    for (int order = 0; order < d; ++order) {
        if (result.size() < 2) {
            return {};
        }
        std::vector<double> diff;
        diff.reserve(result.size() - 1);
        for (std::size_t i = 1; i < result.size(); ++i) {
            diff.push_back(result[i] - result[i - 1]);
        }
        result = std::move(diff);
    }

    return result;
}

}  // namespace ag::util
