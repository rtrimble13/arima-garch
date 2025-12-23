#include "ag/simulation/Innovations.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::simulation {

Innovations::Innovations(unsigned int seed)
    : rng_(seed), normal_(0.0, 1.0), student_t_(1.0) {  // Initialize with df=1, will be updated
}

double Innovations::drawNormal() {
    return normal_(rng_);
}

double Innovations::drawStudentT(double df) {
    if (df <= 2.0) {
        throw std::invalid_argument(
            "Degrees of freedom must be > 2 for Student-t with finite variance");
    }

    // Update the distribution parameters
    student_t_ = std::student_t_distribution<double>(df);

    // Draw from Student-t and standardize to unit variance
    double t_raw = student_t_(rng_);

    // Student-t with df degrees of freedom has variance df/(df-2)
    // Scale to unit variance: z = t / sqrt(df/(df-2))
    double scale = std::sqrt(df / (df - 2.0));
    return t_raw / scale;
}

void Innovations::reseed(unsigned int seed) {
    rng_.seed(seed);
    // Reset the distribution states to ensure reproducibility
    normal_.reset();
    student_t_.reset();
}

}  // namespace ag::simulation
