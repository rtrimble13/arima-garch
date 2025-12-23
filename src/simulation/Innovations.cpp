#include "ag/simulation/Innovations.hpp"

#include <stdexcept>

namespace ag::simulation {

Innovations::Innovations(unsigned int seed) : rng_(seed), normal_(0.0, 1.0) {}

double Innovations::drawNormal() {
    return normal_(rng_);
}

double Innovations::drawStudentT(double /*df*/) {
    throw std::runtime_error("Student-t innovations not yet implemented");
}

void Innovations::reseed(unsigned int seed) {
    rng_.seed(seed);
    // Reset the distribution state to ensure reproducibility
    normal_.reset();
}

}  // namespace ag::simulation
