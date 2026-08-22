#pragma once

#include "cme/generator.hpp"
#include <span>

namespace cme {

struct ReversibleGenerator {
    RestrictedGenerator restricted;
    std::vector<double> stationary_sqrt;
    num::Matrix symmetric_operator;
};

/// Check detailed balance and attach a symmetric representation.
[[nodiscard]] ReversibleGenerator reversible_generator(RestrictedGenerator restricted,
                                                       std::span<const double> stationary_sqrt);

/// A symmetric Laplacian and weights defining its reversible generator.
class ReversibleLaplacian {
  public:
    ReversibleLaplacian(const num::SparseMatrix &laplacian,
                        const std::vector<double> &stationary_sqrt);

    [[nodiscard]] const num::SparseMatrix &matrix() const { return *laplacian_; }
    [[nodiscard]] const std::vector<double> &stationary_sqrt() const { return *stationary_sqrt_; }

  private:
    const num::SparseMatrix *laplacian_;
    const std::vector<double> *stationary_sqrt_;
};

/// Convert a symmetric Laplacian restriction into a Markov generator.
[[nodiscard]] RestrictedGenerator restrict_laplacian(const num::SparseMatrix &laplacian,
                                                     const std::vector<double> &stationary_sqrt,
                                                     const std::vector<num::idx> &states);

} // namespace cme
