#include "ag/data/TimeSeries.hpp"

#include <numeric>

namespace ag::data {

double SeriesView::mean() const {
    if (size_ == 0) {
        return 0.0;
    }
    double sum = std::accumulate(data_, data_ + size_, 0.0);
    return sum / static_cast<double>(size_);
}

double TimeSeries::mean() const {
    if (data_.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(data_.begin(), data_.end(), 0.0);
    return sum / static_cast<double>(data_.size());
}

}  // namespace ag::data
