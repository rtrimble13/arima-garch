#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace ag::data {

// Forward declarations
class TimeSeries;

/**
 * @brief Lightweight view over a contiguous sequence of time series data.
 *
 * SeriesView provides a non-owning reference to time series data, similar to std::span.
 * It allows efficient access to subsequences without copying data.
 */
class SeriesView {
public:
    /**
     * @brief Construct a view from raw pointer and size.
     * @param data Pointer to the first element
     * @param size Number of elements in the view
     */
    constexpr SeriesView(const double* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    /**
     * @brief Construct a view from a std::span.
     * @param span The span to create a view from
     */
    constexpr SeriesView(std::span<const double> span) noexcept
        : data_(span.data()), size_(span.size()) {}

    /**
     * @brief Construct a view from a vector.
     * @param vec The vector to create a view from
     */
    SeriesView(const std::vector<double>& vec) noexcept : data_(vec.data()), size_(vec.size()) {}

    /**
     * @brief Get the number of elements in the view.
     * @return The number of elements
     */
    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }

    /**
     * @brief Check if the view is empty.
     * @return true if the view contains no elements, false otherwise
     */
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    /**
     * @brief Access element at the specified index.
     * @param idx Index of the element
     * @return The element at the specified index
     */
    [[nodiscard]] constexpr double operator[](std::size_t idx) const noexcept { return data_[idx]; }

    /**
     * @brief Get a pointer to the underlying data.
     * @return Pointer to the first element
     */
    [[nodiscard]] constexpr const double* data() const noexcept { return data_; }

    /**
     * @brief Calculate the mean of the values in the view.
     * @return The arithmetic mean of all values
     */
    [[nodiscard]] double mean() const;

    /**
     * @brief Get iterator to the beginning.
     * @return Pointer to the first element
     */
    [[nodiscard]] constexpr const double* begin() const noexcept { return data_; }

    /**
     * @brief Get iterator to the end.
     * @return Pointer to one past the last element
     */
    [[nodiscard]] constexpr const double* end() const noexcept { return data_ + size_; }

private:
    const double* data_;
    std::size_t size_;
};

/**
 * @brief A container for time series data.
 *
 * TimeSeries stores a sequence of double-precision values representing observations
 * over time. It provides basic statistical operations and efficient views.
 */
class TimeSeries {
public:
    /**
     * @brief Default constructor creates an empty time series.
     */
    TimeSeries() = default;

    /**
     * @brief Construct a time series from a vector of values.
     * @param values The time series data
     */
    explicit TimeSeries(std::vector<double> values) : data_(std::move(values)) {}

    /**
     * @brief Construct a time series from an initializer list.
     * @param values The time series data
     */
    TimeSeries(std::initializer_list<double> values) : data_(values) {}

    /**
     * @brief Get the number of observations in the time series.
     * @return The number of observations
     */
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

    /**
     * @brief Check if the time series is empty.
     * @return true if the time series contains no observations, false otherwise
     */
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    /**
     * @brief Access observation at the specified index.
     * @param idx Index of the observation
     * @return The observation at the specified index
     */
    [[nodiscard]] double operator[](std::size_t idx) const noexcept { return data_[idx]; }

    /**
     * @brief Access observation at the specified index.
     * @param idx Index of the observation
     * @return Reference to the observation at the specified index
     */
    [[nodiscard]] double& operator[](std::size_t idx) noexcept { return data_[idx]; }

    /**
     * @brief Get a pointer to the underlying data.
     * @return Pointer to the first element
     */
    [[nodiscard]] const double* data() const noexcept { return data_.data(); }

    /**
     * @brief Get a pointer to the underlying data.
     * @return Pointer to the first element
     */
    [[nodiscard]] double* data() noexcept { return data_.data(); }

    /**
     * @brief Calculate the mean of the observations.
     * @return The arithmetic mean of all observations
     */
    [[nodiscard]] double mean() const;

    /**
     * @brief Create a view of the entire time series.
     * @return A SeriesView spanning all observations
     */
    [[nodiscard]] SeriesView view() const noexcept {
        return SeriesView(data_.data(), data_.size());
    }

    /**
     * @brief Create a view of a subsequence of the time series.
     * @param start Starting index (inclusive)
     * @param count Number of elements in the view
     * @return A SeriesView spanning the specified subsequence
     * @note If start + count exceeds the size, count is adjusted to fit within bounds
     */
    [[nodiscard]] SeriesView view(std::size_t start, std::size_t count) const noexcept {
        // Clamp start to valid range
        start = std::min(start, data_.size());
        // Clamp count to not exceed available elements
        count = std::min(count, data_.size() - start);
        return SeriesView(data_.data() + start, count);
    }

    /**
     * @brief Get iterator to the beginning.
     * @return Iterator to the first element
     */
    [[nodiscard]] auto begin() noexcept { return data_.begin(); }

    /**
     * @brief Get const iterator to the beginning.
     * @return Const iterator to the first element
     */
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }

    /**
     * @brief Get iterator to the end.
     * @return Iterator to one past the last element
     */
    [[nodiscard]] auto end() noexcept { return data_.end(); }

    /**
     * @brief Get const iterator to the end.
     * @return Const iterator to one past the last element
     */
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

private:
    std::vector<double> data_;
};

}  // namespace ag::data
