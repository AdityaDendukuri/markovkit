// a chemical reaction network, the markov_model of the chemical master equation
#pragma once

#include "markovkit/chain.hpp"
#include <functional>
#include <utility>

namespace markovkit {

// x becomes x + change at rate(x); capture rate constants in the lambda
struct reaction {
    state change;
    std::function<real(const state &)> rate;
};

struct reaction_network {
    array<reaction> reactions;

    // every reaction with a positive rate whose result has no negative count
    [[nodiscard]] array<transition> transitions(const state &x) const {
        array<transition> result;
        for (const reaction &r : reactions) {
            const real rate = r.rate(x);
            if (!(rate > 0.0)) {
                continue;
            }
            state y = x;
            bool valid = true;
            for (idx s = 0; s < y.size(); ++s) {
                y[s] += r.change[s];
                valid = valid && y[s] >= 0;
            }
            if (valid) {
                result.push_back({std::move(y), rate});
            }
        }
        return result;
    }
};

} // namespace markovkit
