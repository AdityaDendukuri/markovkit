/// @file rdme/shifted_generator.hpp
/// @brief Matrix-free shifted generator operator for resolvent and implicit solves.
#pragma once

#include "container/vector.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rdme {

/// Matrix-free operator A = shift I - Q retaining a live reference to Q.
/// Rate refreshes on the underlying generator are therefore visible immediately.
template <class Generator>
class shifted_generator {
  public:
    using domain_type = num::vec<double>;
    using codomain_type = num::vec<double>;

    shifted_generator(const Generator &generator, double shift)
        : generator_(&generator), shift_(shift), generator_image_(generator.rows(), 0.0) {
        if (!(shift_ > 0.0) || !std::isfinite(shift_)) {
            throw std::invalid_argument("RDME generator shift must be finite and positive");
        }
    }

    [[nodiscard]] num::idx rows() const noexcept { return generator_->rows(); }
    [[nodiscard]] num::idx cols() const noexcept { return generator_->cols(); }
    [[nodiscard]] double shift() const noexcept { return shift_; }
    [[nodiscard]] const Generator &generator() const noexcept { return *generator_; }

    void apply(const num::vec<double> &x, num::vec<double> &y) const {
        if (x.size() != rows() || y.size() != rows()) {
            throw std::invalid_argument("RDME shifted generator vector size mismatch");
        }
        generator_->apply(x, generator_image_);
        for (num::idx state = 0; state < rows(); ++state) {
            y[state] = shift_ * x[state] - generator_image_[state];
        }
    }

  private:
    const Generator *generator_;
    double shift_;
    mutable num::vec<double> generator_image_;
};

template <class Generator>
shifted_generator(const Generator &, double) -> shifted_generator<Generator>;

} // namespace rdme
