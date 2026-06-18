#include "ag/util/Differencing.hpp"

#include <stdexcept>
#include <utility>

namespace ag::util {

std::vector<double> differenceSeries(const double* data, std::size_t size, int d) {
    if (data == nullptr && size > 0) {
        throw std::invalid_argument("differenceSeries: data is null but size > 0");
    }
    if (d < 0) {
        throw std::invalid_argument("differenceSeries: d must be >= 0");
    }
    if (size == 0) {
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

StreamingDifferencer::StreamingDifferencer(int d) : d_(d) {
    if (d < 0) {
        throw std::invalid_argument("StreamingDifferencer: d must be >= 0");
    }
    last_.assign(static_cast<std::size_t>(d), 0.0);
}

bool StreamingDifferencer::difference(double level, double& differenced) {
    if (d_ == 0) {
        differenced = level;
        return true;
    }

    // cur[k] is the k-th order difference of the current observation. cur[k]
    // is computable only once we have a previous (k-1)-th order difference,
    // i.e. after at least k prior observations.
    std::vector<double> cur(static_cast<std::size_t>(d_) + 1);
    cur[0] = level;
    for (int k = 1; k <= d_; ++k) {
        if (count_ < static_cast<std::size_t>(k)) {
            // Still priming: store the orders we could compute and bail out.
            for (int j = 0; j < k; ++j) {
                last_[static_cast<std::size_t>(j)] = cur[static_cast<std::size_t>(j)];
            }
            ++count_;
            return false;
        }
        cur[static_cast<std::size_t>(k)] =
            cur[static_cast<std::size_t>(k - 1)] - last_[static_cast<std::size_t>(k - 1)];
    }

    // Fully primed: advance the anchors for orders 0..d-1 and emit Δ^d.
    for (int j = 0; j < d_; ++j) {
        last_[static_cast<std::size_t>(j)] = cur[static_cast<std::size_t>(j)];
    }
    ++count_;
    differenced = cur[static_cast<std::size_t>(d_)];
    return true;
}

double StreamingDifferencer::integrate(double differenced) {
    if (d_ == 0) {
        return differenced;
    }

    std::vector<double> cur(static_cast<std::size_t>(d_) + 1);
    cur[static_cast<std::size_t>(d_)] = differenced;
    for (int k = d_ - 1; k >= 0; --k) {
        cur[static_cast<std::size_t>(k)] =
            last_[static_cast<std::size_t>(k)] + cur[static_cast<std::size_t>(k + 1)];
    }
    for (int k = 0; k < d_; ++k) {
        last_[static_cast<std::size_t>(k)] = cur[static_cast<std::size_t>(k)];
    }
    return cur[0];
}

bool StreamingDifferencer::primed() const noexcept {
    return count_ >= static_cast<std::size_t>(d_);
}

}  // namespace ag::util
