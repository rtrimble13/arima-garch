#include "ag/estimation/ParameterVector.hpp"

#include <stdexcept>
#include <string>

namespace ag::estimation::param_vector {

std::size_t size(const ag::models::ArimaGarchSpec& spec) noexcept {
    std::size_t n = 0;
    if (!spec.arimaSpec.isZeroOrder()) {
        n += 1;  // intercept
        n += static_cast<std::size_t>(spec.arimaSpec.p);
        n += static_cast<std::size_t>(spec.arimaSpec.q);
    }
    if (!spec.garchSpec.isNull()) {
        n += 1;                                           // omega
        n += static_cast<std::size_t>(spec.garchSpec.q);  // alpha
        n += static_cast<std::size_t>(spec.garchSpec.p);  // beta
    }
    return n;
}

std::vector<double> pack(const ag::models::ArimaGarchSpec& spec,
                         const ag::models::arima::ArimaParameters& arima_params,
                         const ag::models::garch::GarchParameters& garch_params) {
    std::vector<double> out;
    out.reserve(size(spec));

    if (!spec.arimaSpec.isZeroOrder()) {
        out.push_back(arima_params.intercept);
        for (int i = 0; i < spec.arimaSpec.p; ++i) {
            out.push_back(arima_params.ar_coef[i]);
        }
        for (int i = 0; i < spec.arimaSpec.q; ++i) {
            out.push_back(arima_params.ma_coef[i]);
        }
    }

    if (!spec.garchSpec.isNull()) {
        out.push_back(garch_params.omega);
        for (int i = 0; i < spec.garchSpec.q; ++i) {
            out.push_back(garch_params.alpha_coef[i]);
        }
        for (int i = 0; i < spec.garchSpec.p; ++i) {
            out.push_back(garch_params.beta_coef[i]);
        }
    }

    return out;
}

void unpack(const std::vector<double>& params, const ag::models::ArimaGarchSpec& spec,
            ag::models::arima::ArimaParameters& out_arima,
            ag::models::garch::GarchParameters& out_garch) {
    const std::size_t expected = size(spec);
    if (params.size() != expected) {
        throw std::invalid_argument("param_vector::unpack: vector has " +
                                    std::to_string(params.size()) + " entries but spec requires " +
                                    std::to_string(expected));
    }

    std::size_t idx = 0;

    if (!spec.arimaSpec.isZeroOrder()) {
        out_arima.intercept = params[idx++];
        for (int i = 0; i < spec.arimaSpec.p; ++i) {
            out_arima.ar_coef[i] = params[idx++];
        }
        for (int i = 0; i < spec.arimaSpec.q; ++i) {
            out_arima.ma_coef[i] = params[idx++];
        }
    } else {
        out_arima.intercept = 0.0;
    }

    if (!spec.garchSpec.isNull()) {
        out_garch.omega = params[idx++];
        for (int i = 0; i < spec.garchSpec.q; ++i) {
            out_garch.alpha_coef[i] = params[idx++];
        }
        for (int i = 0; i < spec.garchSpec.p; ++i) {
            out_garch.beta_coef[i] = params[idx++];
        }
    }
}

}  // namespace ag::estimation::param_vector
