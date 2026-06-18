#include "ag/stats/ACF.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace ag::stats {

namespace {

// Below this length the direct O(n·max_lag) computation is used as a simple,
// exact reference; at or above it the FFT path (O(n log n)) wins.
constexpr std::size_t FFT_THRESHOLD = 64;

// In-place iterative radix-2 Cooley-Tukey FFT. `a.size()` must be a power of 2.
// `invert == true` computes the inverse transform (scaled by 1/n).
void fft(std::vector<std::complex<double>>& a, bool invert) {
    const std::size_t n = a.size();

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang =
            2.0 * std::numbers::pi / static_cast<double>(len) * (invert ? 1.0 : -1.0);
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (auto& x : a) {
            x /= static_cast<double>(n);
        }
    }
}

// Linear autocovariance sums Σ_i x_i·x_{i+k} for k = 0..max_lag, computed via
// the Wiener-Khinchin theorem: zero-pad to length >= 2n-1 (a power of 2),
// transform, take the power spectrum, and inverse-transform. The zero padding
// makes the circular correlation equal the linear one, so the result matches
// the direct sum to floating-point precision.
std::vector<double> autocovariance_sums_fft(const std::vector<double>& demeaned,
                                            std::size_t max_lag) {
    const std::size_t n = demeaned.size();

    std::size_t m = 1;
    while (m < 2 * n) {
        m <<= 1;
    }

    std::vector<std::complex<double>> fa(m, std::complex<double>(0.0, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        fa[i] = std::complex<double>(demeaned[i], 0.0);
    }

    fft(fa, false);
    for (auto& v : fa) {
        v = std::complex<double>(std::norm(v), 0.0);  // |X(f)|^2
    }
    fft(fa, true);

    std::vector<double> gamma(max_lag + 1);
    for (std::size_t k = 0; k <= max_lag; ++k) {
        gamma[k] = fa[k].real();
    }
    return gamma;
}

}  // namespace

double acf_at_lag(std::span<const double> data, std::size_t lag) {
    const std::size_t n = data.size();

    if (n == 0) {
        throw std::invalid_argument("Cannot compute ACF of empty data");
    }

    if (lag >= n) {
        throw std::invalid_argument("Lag must be less than data size");
    }

    // Special case: ACF at lag 0 is always 1
    if (lag == 0) {
        return 1.0;
    }

    // Compute mean
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / static_cast<double>(n);

    // Compute variance (denominator) using all data points
    double variance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }

    // Handle constant series
    if (variance == 0.0) {
        return 0.0;
    }

    // Compute autocovariance at lag k (numerator)
    double autocovariance = 0.0;
    for (std::size_t i = 0; i < n - lag; ++i) {
        autocovariance += (data[i] - mean) * (data[i + lag] - mean);
    }

    // ACF = autocovariance / variance
    return autocovariance / variance;
}

std::vector<double> acf(std::span<const double> data, std::size_t max_lag) {
    const std::size_t n = data.size();

    if (n == 0) {
        throw std::invalid_argument("Cannot compute ACF of empty data");
    }

    if (max_lag >= n) {
        throw std::invalid_argument("max_lag must be less than data size");
    }

    std::vector<double> result;
    result.reserve(max_lag + 1);

    // Compute mean once
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / static_cast<double>(n);

    // Demean once and reuse for the variance and every lag.
    std::vector<double> demeaned(n);
    double variance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        demeaned[i] = data[i] - mean;
        variance += demeaned[i] * demeaned[i];
    }

    // Handle constant series - all ACF values are 0 except lag 0
    if (variance == 0.0) {
        result.push_back(1.0);  // ACF at lag 0
        for (std::size_t lag = 1; lag <= max_lag; ++lag) {
            result.push_back(0.0);
        }
        return result;
    }

    // Lag 0: ACF is always 1.0
    result.push_back(1.0);

    if (n >= FFT_THRESHOLD) {
        // FFT path: O(n log n) autocovariances (Wiener-Khinchin).
        std::vector<double> gamma = autocovariance_sums_fft(demeaned, max_lag);
        for (std::size_t lag = 1; lag <= max_lag; ++lag) {
            result.push_back(gamma[lag] / variance);
        }
        return result;
    }

    // Direct path (reference for small n): O(n·max_lag).
    for (std::size_t lag = 1; lag <= max_lag; ++lag) {
        double autocovariance = 0.0;
        for (std::size_t i = 0; i < n - lag; ++i) {
            autocovariance += demeaned[i] * demeaned[i + lag];
        }
        result.push_back(autocovariance / variance);
    }

    return result;
}

}  // namespace ag::stats
