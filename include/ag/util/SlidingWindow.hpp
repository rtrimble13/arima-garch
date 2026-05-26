#pragma once

#include <algorithm>
#include <vector>

namespace ag::util {

/**
 * @brief Shift the elements of buf left by one and write @p value into the
 *        last slot.
 *
 * Used by ARIMA and GARCH state classes to maintain bounded-history buffers
 * during recursion. No-op on empty buffers.
 */
template <typename T>
void shiftAndAppend(std::vector<T>& buf, const T& value) {
    if (buf.empty()) {
        return;
    }
    std::shift_left(buf.begin(), buf.end(), 1);
    buf.back() = value;
}

}  // namespace ag::util
