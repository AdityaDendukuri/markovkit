#include <markovkit.hpp>
#include <iostream>
#include <cassert>

void test_reaction_system() {
    using namespace fsp;
    
    // Constitutive gene expression model:
    // DNA -> DNA + mRNA (rate k_trans)
    // mRNA -> 0        (rate gamma)
    ReactionSystem sys;
    sys.stoich_vecs = {MultiIndex{1}, MultiIndex{-1}};
    sys.propensities = {
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[0]; },
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[1] * x[0]; }
    };

    assert(sys.n_reactions() == 2);
    std::cout << "[PASS] FSP ReactionSystem initialization\n";
}

void test_state_space() {
    using namespace fsp;
    
    ReactionSystem sys;
    sys.stoich_vecs = {MultiIndex{1}, MultiIndex{-1}};
    sys.propensities = {
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[0]; },
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[1] * x[0]; }
    };

    StateSpace states;
    states.add_state(MultiIndex{0}, 1.0);

    BoundaryCondition bc = rect_boundary(50);

    // Expand state space
    states.expand(sys, bc, 2);
    assert(states.size() >= 2);
    std::cout << "[PASS] FSP StateSpace expansion (" << states.size() << " states)\n";
}

int main() {
    std::cout << "--- Running FSP Tests ---\n";
    test_reaction_system();
    test_state_space();
    std::cout << "All FSP tests passed successfully!\n";
    return 0;
}
