#pragma once

#include "cme/generator.hpp"
#include "core/types.hpp"

namespace cme {

struct reversible_generator_result {
    restricted_generator restricted;
    num::vec stationary_sqrt;
    num::mat symmetric_operator;
};

/// Check detailed balance and attach a symmetric representation.
[[nodiscard]] reversible_generator_result
reversible_generator(restricted_generator restricted, num::view<const double> stationary_sqrt);

/// A symmetric Laplacian and weights defining its reversible generator.
/// Both are referenced, not owned; they must outlive this object.
class reversible_laplacian {
  public:
    reversible_laplacian(const num::spmat &laplacian, num::view<const double> stationary_sqrt);

    [[nodiscard]] const num::spmat &matrix() const { return *laplacian_; }
    [[nodiscard]] num::view<const double> stationary_sqrt() const { return stationary_sqrt_; }

  private:
    const num::spmat *laplacian_;
    num::view<const double> stationary_sqrt_;
};

/// Convert a symmetric Laplacian restriction into a Markov generator.
[[nodiscard]] restricted_generator restrict_laplacian(const num::spmat &laplacian,
                                                      num::view<const double> stationary_sqrt,
                                                      num::view<const num::idx> states);

} // namespace cme
