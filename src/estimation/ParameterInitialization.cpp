#include "ag/estimation/ParameterInitialization.hpp"

#include "ag/stats/ACF.hpp"
#include "ag/stats/Descriptive.hpp"
#include "ag/stats/PACF.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ag::estimation {

namespace {

// Compute sample mean
double computeMean(const double* data, std::size_t size) {
    if (size == 0) {
        return 0.0;
    }
    double sum = std::accumulate(data, data + size, 0.0);
    return sum / static_cast<double>(size);
}

// Compute sample variance
double computeVariance(const double* data, std::size_t size) {
    if (size < 2) {
        return 1.0;  // Fallback for very small samples
    }
    double mean = computeMean(data, size);
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(size - 1);
}

// Apply differencing to data
std::vector<double> difference(const double* data, std::size_t size, int d) {
    std::vector<double> result(data, data + size);

    for (int order = 0; order < d; ++order) {
        if (result.size() < 2) {
            break;  // Can't difference anymore
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

}  // namespace

ag::models::arima::ArimaParameters initializeArimaParameters(const double* data, std::size_t size,
                                                             const ag::models::ArimaSpec& spec) {
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer is null");
    }
    if (size < 10) {
        throw std::invalid_argument(
            "Insufficient data for parameter initialization (need at least 10)");
    }

    const int p = spec.p;
    const int d = spec.d;
    const int q = spec.q;

    ag::models::arima::ArimaParameters params(p, q);

    // Apply differencing if needed
    std::vector<double> working_data;
    if (d > 0) {
        working_data = difference(data, size, d);
        if (working_data.size() < 10) {
            throw std::invalid_argument("Insufficient data after differencing");
        }
    } else {
        working_data.assign(data, data + size);
    }

    // Initialize intercept to sample mean
    params.intercept = computeMean(working_data.data(), working_data.size());

    // Initialize AR coefficients from PACF if p > 0
    if (p > 0) {
        std::size_t max_lag = std::min(static_cast<std::size_t>(p), working_data.size() / 4);
        if (max_lag > 0) {
            try {
                std::vector<double> pacf_values = ag::stats::pacf(working_data, max_lag);
                for (int i = 0; i < p && i < static_cast<int>(pacf_values.size()); ++i) {
                    // Scale by 0.9 to promote stability
                    params.ar_coef[i] = 0.9 * pacf_values[i];
                }
            } catch (...) {
                // If PACF computation fails, use small default values
                for (int i = 0; i < p; ++i) {
                    params.ar_coef[i] = 0.1 / (i + 1);
                }
            }
        } else {
            // Fallback: small positive values
            for (int i = 0; i < p; ++i) {
                params.ar_coef[i] = 0.1 / (i + 1);
            }
        }
    }

    // Initialize MA coefficients from ACF if q > 0
    if (q > 0) {
        std::size_t max_lag = std::min(static_cast<std::size_t>(q), working_data.size() / 4);
        if (max_lag > 0) {
            try {
                std::vector<double> acf_values = ag::stats::acf(working_data, max_lag);
                // Start from lag 1 (skip lag 0 which is always 1.0)
                for (int i = 0; i < q && (i + 1) < static_cast<int>(acf_values.size()); ++i) {
                    // Use negative ACF values scaled by 0.9
                    params.ma_coef[i] = -0.9 * acf_values[i + 1];
                }
            } catch (...) {
                // If ACF computation fails, use small default values
                for (int i = 0; i < q; ++i) {
                    params.ma_coef[i] = 0.1 / (i + 1);
                }
            }
        } else {
            // Fallback: small positive values
            for (int i = 0; i < q; ++i) {
                params.ma_coef[i] = 0.1 / (i + 1);
            }
        }
    }

    return params;
}

ag::models::garch::GarchParameters initializeGarchParameters(const double* residuals,
                                                             std::size_t size,
                                                             const ag::models::GarchSpec& spec) {
    if (residuals == nullptr) {
        throw std::invalid_argument("Residuals pointer is null");
    }
    if (size < 10) {
        throw std::invalid_argument(
            "Insufficient residuals for parameter initialization (need at least 10)");
    }

    const int p = spec.p;
    const int q = spec.q;

    ag::models::garch::GarchParameters params(p, q);

    // Compute sample variance of residuals
    double sample_var = computeVariance(residuals, size);
    if (sample_var <= 0.0) {
        sample_var = 1.0;  // Fallback to avoid issues
    }

    // Target persistence: sum(alpha) + sum(beta) ≈ 0.90
    // This gives reasonable volatility clustering while maintaining stationarity
    double target_persistence = 0.90;

    // Allocate persistence between ARCH (alpha) and GARCH (beta) effects
    // Typically GARCH effects are stronger, so use 30% for ARCH, 70% for GARCH
    double alpha_total = (q > 0) ? target_persistence * 0.30 : 0.0;
    double beta_total = (p > 0) ? target_persistence * 0.70 : 0.0;

    // If only ARCH or only GARCH, adjust accordingly
    if (q > 0 && p == 0) {
        alpha_total = target_persistence;
    } else if (p > 0 && q == 0) {
        beta_total = target_persistence;
    }

    // Distribute alpha coefficients evenly
    if (q > 0) {
        double alpha_each = alpha_total / q;
        for (int i = 0; i < q; ++i) {
            params.alpha_coef[i] = alpha_each;
        }
    }

    // Distribute beta coefficients with decay (first coefficient is largest)
    if (p > 0) {
        double sum_weights = 0.0;
        for (int i = 0; i < p; ++i) {
            sum_weights += 1.0 / (i + 1);
        }
        for (int i = 0; i < p; ++i) {
            params.beta_coef[i] = beta_total * (1.0 / (i + 1)) / sum_weights;
        }
    }

    // Set omega based on unconditional variance formula:
    // sigma^2 = omega / (1 - sum(alpha) - sum(beta))
    // => omega = sigma^2 * (1 - sum(alpha) - sum(beta))
    double persistence = 0.0;
    for (int i = 0; i < q; ++i) {
        persistence += params.alpha_coef[i];
    }
    for (int i = 0; i < p; ++i) {
        persistence += params.beta_coef[i];
    }

    params.omega = sample_var * (1.0 - persistence);

    // Ensure omega is positive (should be by construction, but just in case)
    if (params.omega <= 0.0) {
        params.omega = 0.01 * sample_var;
    }

    return params;
}

std::pair<ag::models::arima::ArimaParameters, ag::models::garch::GarchParameters>
initializeArimaGarchParameters(const double* data, std::size_t size,
                               const ag::models::ArimaGarchSpec& spec) {
    // Initialize ARIMA parameters
    auto arima_params = initializeArimaParameters(data, size, spec.arimaSpec);

    // Compute residuals using initialized ARIMA parameters
    ag::models::arima::ArimaModel arima_model(spec.arimaSpec);
    std::vector<double> residuals = arima_model.computeResiduals(data, size, arima_params);

    if (residuals.empty()) {
        throw std::invalid_argument("Failed to compute residuals for GARCH initialization");
    }

    // Initialize GARCH parameters from residuals
    auto garch_params =
        initializeGarchParameters(residuals.data(), residuals.size(), spec.garchSpec);

    return {arima_params, garch_params};
}

std::vector<double> perturbParameters(const std::vector<double>& params, double scale,
                                      std::mt19937& rng) {
    std::vector<double> perturbed = params;
    std::normal_distribution<double> dist(0.0, 1.0);

    for (std::size_t i = 0; i < perturbed.size(); ++i) {
        // For each parameter, add noise proportional to its magnitude
        // Use scale * |param| as standard deviation, with minimum of 0.01
        double param_scale = std::max(std::abs(params[i]), 0.01);
        double noise = scale * param_scale * dist(rng);
        perturbed[i] += noise;
    }

    return perturbed;
}

}  // namespace ag::estimation
