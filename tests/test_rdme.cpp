#include "linear/solvers/gmres.hpp"
#include "rdme/rdme.hpp"
#include "structures/graph/graph.hpp"
#include "structures/graph/structured_grid.hpp"
#include "check.hpp"
#include <cmath>
#include <iostream>
#include <numeric>

void test_structured_voxel_domain() {
    num::structured_grid_graph<2> grid({3, 2}, {0.5, 2.0});
    rdme::voxel_domain domain(std::move(grid), num::array<double>{0.25, 1.0});

    check::that(domain.n_voxels() == 6, "domain.n_voxels() == 6");
    check::that(domain.n_species() == 2, "domain.n_species() == 2");
    check::that(std::abs(domain.voxel_volume(0) - 1.0) < 1e-15, "std::abs(domain.voxel_volume(0) - 1.0) < 1e-15");

    const auto neighbors = domain.graph().neighbors(0);
    check::that(neighbors.size() == 2, "neighbors.size() == 2");
    check::that(std::abs(domain.jump_rate(0, 0, neighbors.begin()[0]) - 1.0) < 1e-15, "std::abs(domain.jump_rate(0, 0, neighbors.begin()[0]) - 1.0) < 1e-15");
    check::that(std::abs(domain.jump_rate(0, 0, neighbors.begin()[1]) - 0.0625) < 1e-15, "std::abs(domain.jump_rate(0, 0, neighbors.begin()[1]) - 0.0625) < 1...");
    check::that(std::abs(domain.outgoing_jump_rate(0, 0) - 1.0625) < 1e-15, "std::abs(domain.outgoing_jump_rate(0, 0) - 1.0625) < 1e-15");
}

void test_explicit_graph_voxel_domain() {
    num::graph<double, num::idx> graph(2);
    graph.add_edge(0, 1, 3.0);
    rdme::voxel_domain domain(std::move(graph), num::array<double>{2.0, 4.0},
                              num::array<double>{0.5});

    const auto edge_from_zero = domain.graph().neighbors(0)[0];
    const auto edge_from_one = domain.graph().neighbors(1)[0];
    check::that(std::abs(domain.jump_rate(0, 0, edge_from_zero) - 0.75) < 1e-15, "std::abs(domain.jump_rate(0, 0, edge_from_zero) - 0.75) < 1e-15");
    check::that(std::abs(domain.jump_rate(0, 1, edge_from_one) - 0.375) < 1e-15, "std::abs(domain.jump_rate(0, 1, edge_from_one) - 0.375) < 1e-15");
}

void test_exact_cme_lift() {
    num::structured_grid_graph<1> grid({2}, {0.5});
    rdme::voxel_domain domain(std::move(grid), num::array<double>{1.0});

    cme::reaction_system local;
    local.changes = {{1}};
    local.propensities = {
        [](const markovkit::state &, const num::array<double> &rates, double) { return rates[0]; }};

    const auto lifted = rdme::lift_to_cme(domain, local);
    check::that(lifted.size() == 4, "lifted.size() == 4"); // two local births and two directed diffusion jumps
    check::that(lifted.changes[0] == markovkit::state({1, 0}), "lifted.changes[0] == markovkit::state({1, 0})");
    check::that(lifted.changes[1] == markovkit::state({0, 1}), "lifted.changes[1] == markovkit::state({0, 1})");
    check::that(lifted.changes[2] == markovkit::state({-1, 1}), "lifted.changes[2] == markovkit::state({-1, 1})");
    check::that(lifted.changes[3] == markovkit::state({1, -1}), "lifted.changes[3] == markovkit::state({1, -1})");

    const markovkit::state state{3, 2};
    const num::array<double> rates{0.25};
    check::that(std::abs(lifted.propensities[0](state, rates, 0.0) - 0.25) < 1e-15, "std::abs(lifted.propensities[0](state, rates, 0.0) - 0.25) < 1e-15");
    // h=0.5: transmissibility=2, volume=0.5, hence rate 4 per molecule.
    check::that(std::abs(lifted.propensities[2](state, rates, 0.0) - 12.0) < 1e-15, "std::abs(lifted.propensities[2](state, rates, 0.0) - 12.0) < 1e-15");
    check::that(std::abs(lifted.propensities[3](state, rates, 0.0) - 8.0) < 1e-15, "std::abs(lifted.propensities[3](state, rates, 0.0) - 8.0) < 1e-15");
}

void test_exact_cme_lift_supports_dynamic_states() {
    num::structured_grid_graph<1> grid({9});
    rdme::voxel_domain domain(std::move(grid), num::array<double>{1.0});
    const auto lifted = rdme::lift_to_cme(domain, cme::reaction_system{});
    check::that(!lifted.changes.empty(), "nine-voxel lift has diffusion events");
    check::that(lifted.changes.front().size() == 9, "lift uses dynamic state storage");
}

void test_dynamic_diffusion_sectors_scale_past_cme_limit() {
    rdme::diffusion_state_space k4(3, {4});
    rdme::diffusion_state_space k8(3, {8});
    rdme::diffusion_state_space k16(3, {16});
    rdme::diffusion_state_space k64(3, {64});
    check::that(k4.size() == 15, "k4.size() == 15");
    check::that(k8.size() == 45, "k8.size() == 45");
    check::that(k16.size() == 153, "k16.size() == 153");
    check::that(k64.size() == 2145, "k64.size() == 2145");
    check::that(k16.layout().size() == 3, "k16.layout().size() == 3");

    rdme::diffusion_state_space two_species(2, {1, 2});
    check::that(two_species.size() == 6, "two_species.size() == 6");
    for (const auto &state : two_species.states()) {
        check::that(state[0] + state[2] == 1, "state[0] + state[2] == 1");
        check::that(state[1] + state[3] == 2, "state[1] + state[3] == 2");
    }

    // Arithmetic ranking exactly reproduces enumeration, without a state hash table.
    for (num::idx state = 0; state < two_species.size(); ++state) {
        check::that(two_species.rank(two_species[state]) == state, "two_species.rank(two_species[state]) == state");
        check::that(two_species.unrank(state) == two_species[state], "two_species.unrank(state) == two_species[state]");
    }
    check::that(two_species.find({2, 0, 0, 2}) == rdme::diffusion_state_space::npos, "two_species.find({2, 0, 0, 2}) == rdme::diffusion_state_space::npos");

    rdme::reaction_state_space reaction_space(3, {2, 2});
    for (num::idx state = 0; state < reaction_space.size(); ++state) {
        check::that(reaction_space.find(reaction_space[state]) == state, "reaction_space.find(reaction_space[state]) == state");
        const num::idx sector = reaction_space.sector(state);
        check::that(state >= reaction_space.sector_offsets()[sector], "state >= reaction_space.sector_offsets()[sector]");
        check::that(state < reaction_space.sector_offsets()[sector + 1], "state < reaction_space.sector_offsets()[sector + 1]");
    }
    check::that(reaction_space.find({3, 0, 0, 0, 0, 0}) == rdme::reaction_state_space::npos, "reaction_space.find({3, 0, 0, 0, 0, 0}) == rdme::reaction_state_spa...");
}

void test_cached_matrix_free_diffusion_generator() {
    num::structured_grid_graph<1> grid({3}, {0.5});
    rdme::voxel_domain domain(std::move(grid), num::array<double>{0.25});
    rdme::diffusion_generator generator(std::move(domain), num::array<int>{4});
    static_assert(num::linear_operator<decltype(generator), num::vec<double>, num::vec<double>>);

    check::that(generator.rows() == 15, "generator.rows() == 15");
    const rdme::configuration concentrated{4, 0, 0};
    const num::idx source = generator.state_space().find(concentrated);
    check::that(source != rdme::diffusion_state_space::npos, "source != rdme::diffusion_state_space::npos");

    num::vec<double> probability(generator.rows(), 0.0);
    probability[source] = 1.0;
    num::vec<double> matrix_free(generator.rows());
    generator.apply(probability, matrix_free);

    // h=0.5 gives transmissibility 2 and volume 0.5: D*T/V = 1 per molecule.
    check::that(std::abs(generator.exit_rates()[source] - 4.0) < 1e-15, "std::abs(generator.exit_rates()[source] - 4.0) < 1e-15");
    check::that(std::abs(matrix_free[source] + 4.0) < 1e-15, "std::abs(matrix_free[source] + 4.0) < 1e-15");
    check::that(std::abs(std::accumulate(matrix_free.begin(), matrix_free.end(), 0.0)) < 1e-15, "std::abs(std::accumulate(matrix_free.begin(), matrix_free.end(), 0....");

    const num::spmat sparse = generator.to_sparse();
    num::vec<double> assembled(generator.rows(), 0.0);
    sparse.apply(probability, assembled);
    for (num::idx state = 0; state < generator.rows(); ++state) {
        check::that(std::abs(matrix_free[state] - assembled[state]) < 1e-15, "std::abs(matrix_free[state] - assembled[state]) < 1e-15");
    }

    // Reuse the same output vector with a different vector; apply performs no setup or allocation.
    probability[source] = 0.0;
    probability[generator.state_space().find({0, 0, 4})] = 1.0;
    generator.apply(probability, matrix_free);
    const auto &reused = matrix_free;
    check::that(std::abs(std::accumulate(reused.begin(), reused.end(), 0.0)) < 1e-15, "std::abs(std::accumulate(reused.begin(), reused.end(), 0.0)) < 1e-15");
}

void test_reactions_couple_contiguous_population_sectors() {
    num::structured_grid_graph<1> grid({2}, {1.0});
    rdme::voxel_domain domain(std::move(grid), num::array<double>{0.5});
    num::array<rdme::local_reaction> reactions{
        rdme::local_reaction{{1},
                             [](num::view<const int>, num::view<const double> parameters,
                                double volume, double) { return parameters[0] * volume; }},
        rdme::local_reaction{
            {-1},
            [](num::view<const int> counts, num::view<const double> parameters, double, double) {
                return parameters[1] * static_cast<double>(counts[0]);
            }}};
    rdme::reaction_diffusion_model model(std::move(domain), std::move(reactions));
    rdme::restricted_generator generator(std::move(model), num::array<int>{3},
                                         num::array<double>{2.0, 0.25}, 0.0,
                                         rdme::boundary_mode::absorbing);
    static_assert(num::linear_operator<decltype(generator), num::vec<double>, num::vec<double>>);

    // For two voxels, sector k contains k+1 configurations.
    check::that(generator.rows() == 10, "generator.rows() == 10");
    check::that(generator.state_space().sector_offsets() == num::array<num::idx>({0, 1, 3, 6, 10}), "generator.state_space().sector_offsets() == num::array<num::idx>({0...");

    bool saw_diffusion_block = false;
    bool saw_birth_superblock = false;
    bool saw_death_subblock = false;
    for (num::idx source = 0; source < generator.rows(); ++source) {
        const auto source_sector = generator.state_space().sector(source);
        for (num::idx entry = generator.source_offsets()[source];
             entry < generator.source_offsets()[source + 1]; ++entry) {
            const auto destination_sector =
                generator.state_space().sector(generator.transitions()[entry].destination);
            const auto difference =
                static_cast<long long>(destination_sector) - static_cast<long long>(source_sector);
            check::that(std::abs(difference) <= 1, "std::abs(difference) <= 1");
            saw_diffusion_block = saw_diffusion_block || difference == 0;
            saw_birth_superblock = saw_birth_superblock || difference == 1;
            saw_death_subblock = saw_death_subblock || difference == -1;
        }
    }
    check::that(saw_diffusion_block && saw_birth_superblock && saw_death_subblock, "saw_diffusion_block && saw_birth_superblock && saw_death_subblock");

    const num::idx empty = generator.state_space().find({0, 0});
    const num::idx boundary = generator.state_space().find({3, 0});
    check::that(empty != rdme::reaction_state_space::npos, "empty != rdme::reaction_state_space::npos");
    check::that(boundary != rdme::reaction_state_space::npos, "boundary != rdme::reaction_state_space::npos");
    check::that(std::abs(generator.exit_rates()[empty] - 4.0) < 1e-15, "std::abs(generator.exit_rates()[empty] - 4.0) < 1e-15");
    // 1.5 diffusion + 0.75 death + 4.0 birth loss through the truncation boundary.
    check::that(std::abs(generator.exit_rates()[boundary] - 6.25) < 1e-15, "std::abs(generator.exit_rates()[boundary] - 6.25) < 1e-15");

    num::vec<double> probability(generator.rows(), 0.0);
    probability[boundary] = 1.0;
    num::vec<double> matrix_free(generator.rows());
    generator.apply(probability, matrix_free);
    // Only the two out-of-box birth events lose mass from this absorbing truncation.
    check::that(std::abs(std::accumulate(matrix_free.begin(), matrix_free.end(), 0.0) + 4.0) < 1e-15, "std::abs(std::accumulate(matrix_free.begin(), matrix_free.end(), 0....");

    const num::spmat sparse = generator.to_sparse();
    num::vec<double> assembled(generator.rows(), 0.0);
    sparse.apply(probability, assembled);
    for (num::idx state = 0; state < generator.rows(); ++state) {
        check::that(std::abs(matrix_free[state] - assembled[state]) < 1e-15, "std::abs(matrix_free[state] - assembled[state]) < 1e-15");
    }

    const auto old_offsets = generator.source_offsets();
    const auto old_transition_count = generator.transitions().size();
    const num::array<double> new_parameters{1.0, 0.5};
    generator.refresh(new_parameters, 2.0);
    check::that(generator.source_offsets() == old_offsets, "generator.source_offsets() == old_offsets");
    check::that(generator.transitions().size() == old_transition_count, "generator.transitions().size() == old_transition_count");
    check::that(std::abs(generator.exit_rates()[empty] - 2.0) < 1e-15, "std::abs(generator.exit_rates()[empty] - 2.0) < 1e-15");
    check::that(generator.time() == 2.0, "generator.time() == 2.0");
}

void test_population_conserving_recursive_multigrid() {
    num::structured_grid_graph<1> grid({8}, {1.0});
    rdme::voxel_domain domain(grid, num::array<double>{1.0});
    rdme::population_multigrid multigrid(domain, num::array<int>{2}, 1.0);
    static_assert(num::linear_operator<decltype(multigrid), num::vec<double>, num::vec<double>>);

    check::that(multigrid.rows() == 36, "multigrid.rows() == 36");
    check::that(multigrid.transfer().coarse_size() == 10, "multigrid.transfer().coarse_size() == 10");
    check::that(multigrid.n_levels() == 3, "multigrid.n_levels() == 3");
    check::that(multigrid.grid_complexity() >= 1.0, "multigrid.grid_complexity() >= 1.0");
    check::that(multigrid.operator_complexity() >= 1.0, "multigrid.operator_complexity() >= 1.0");

    // R P is the identity: aggregation exactly reverses the multinomial lift.
    num::vec<double> coarse(multigrid.transfer().coarse_size(), 0.0);
    for (num::idx state = 0; state < coarse.size(); ++state) {
        coarse[state] = std::sin(static_cast<double>(state));
    }
    num::vec<double> fine(multigrid.rows(), 0.0);
    num::vec<double> restricted(coarse.size(), 0.0);
    multigrid.transfer().prolong(coarse, fine);
    multigrid.transfer().restrict(fine, restricted);
    for (num::idx state = 0; state < coarse.size(); ++state) {
        check::that(std::abs(restricted[state] - coarse[state]) < 1e-14, "std::abs(restricted[state] - coarse[state]) < 1e-14");
    }
    check::that(std::abs(std::accumulate(fine.begin(), fine.end(), 0.0) -
                    std::accumulate(coarse.begin(), coarse.end(), 0.0)) < 1e-14, "std::abs(std::accumulate(fine.begin(), fine.end(), 0.0) - std::accu...");

    // Uniform factor-two rediscretization is the Petrov--Galerkin operator R A P.
    num::vec<double> fine_image(multigrid.rows(), 0.0);
    num::vec<double> projected(coarse.size(), 0.0);
    num::vec<double> coarse_image(coarse.size(), 0.0);
    multigrid.apply_system(fine, fine_image);
    multigrid.transfer().restrict(fine_image, projected);
    check::that(multigrid.coarse_level() != nullptr, "multigrid.coarse_level() != nullptr");
    multigrid.coarse_level()->apply_system(coarse, coarse_image);
    for (num::idx state = 0; state < coarse.size(); ++state) {
        check::that(std::abs(projected[state] - coarse_image[state]) < 1e-12, "std::abs(projected[state] - coarse_image[state]) < 1e-12");
    }

    num::vec<double> rhs(multigrid.rows(), 0.0);
    for (num::idx state = 0; state < rhs.size(); ++state) {
        rhs[state] = 0.25 + std::sin(static_cast<double>(state));
    }
    num::vec<double> correction(multigrid.rows(), 0.0);
    num::vec<double> image(multigrid.rows(), 0.0);
    multigrid.apply(rhs, correction);
    multigrid.apply_system(correction, image);
    double initial_norm_squared = 0.0;
    double residual_norm_squared = 0.0;
    for (num::idx state = 0; state < rhs.size(); ++state) {
        initial_norm_squared += rhs[state] * rhs[state];
        const double residual = rhs[state] - image[state];
        residual_norm_squared += residual * residual;
    }
    check::that(residual_norm_squared < initial_norm_squared, "residual_norm_squared < initial_norm_squared");

    // Aggregate-block relaxation exactly couples fine configurations that differ only by
    // particle redistribution inside one coarse voxel. It should improve this representative
    // cycle over point Jacobi while preserving the same hierarchy and transfer operators.
    rdme::population_multigrid block_multigrid(
        domain, num::array<int>{2}, 1.0, 2, 2, 2.0 / 3.0, 4, 32, 1, 0,
        rdme::population_smoother::aggregate_block);
    num::vec<double> block_correction(block_multigrid.rows(), 0.0);
    num::vec<double> block_image(block_multigrid.rows(), 0.0);
    block_multigrid.apply(rhs, block_correction);
    block_multigrid.apply_system(block_correction, block_image);
    double block_residual_norm_squared = 0.0;
    for (num::idx state = 0; state < rhs.size(); ++state) {
        const double residual = rhs[state] - block_image[state];
        block_residual_norm_squared += residual * residual;
    }
    check::that(block_residual_norm_squared < residual_norm_squared, "block_residual_norm_squared < residual_norm_squared");

    // The same V-cycle can be applied independently to every reaction sector.
    rdme::reaction_state_space reaction_space(8, {2});
    rdme::sector_diffusion_multigrid sector_multigrid(domain, reaction_space, 1.0);
    static_assert(num::linear_operator<decltype(sector_multigrid), num::vec<double>, num::vec<double>>);
    check::that(sector_multigrid.rows() == 45, "sector_multigrid.rows() == 45");
    check::that(sector_multigrid.n_sectors() == 3, "sector_multigrid.n_sectors() == 3");
    num::vec<double> sector_rhs(sector_multigrid.rows(), 1.0);
    num::vec<double> sector_correction(sector_multigrid.rows(), 0.0);
    sector_multigrid.apply(sector_rhs, sector_correction);
    check::that(std::all_of(sector_correction.begin(), sector_correction.end(),
                       [](double value) { return std::isfinite(value); }), "std::all_of(sector_correction.begin(), sector_correction.end(), [](...");
}

void test_sector_multigrid_preconditions_reaction_diffusion_resolvent() {
    num::structured_grid_graph<1> grid({4}, {1.0});
    rdme::voxel_domain domain(std::move(grid), num::array<double>{4.0});
    num::array<rdme::local_reaction> reactions{
        rdme::local_reaction{{1},
                             [](num::view<const int>, num::view<const double> parameters,
                                double volume, double) { return parameters[0] * volume; }},
        rdme::local_reaction{
            {-1},
            [](num::view<const int> counts, num::view<const double> parameters, double, double) {
                return parameters[1] * static_cast<double>(counts[0]);
            }}};
    rdme::reaction_diffusion_model model(std::move(domain), std::move(reactions));
    rdme::restricted_generator generator(std::move(model), num::array<int>{3},
                                         num::array<double>{0.05, 0.02}, 0.0,
                                         rdme::boundary_mode::absorbing);
    constexpr double shift = 0.5;
    rdme::shifted_generator system(generator, shift);
    rdme::sector_diffusion_multigrid preconditioner(generator.model().domain(),
                                                    generator.state_space(), shift);
    static_assert(num::linear_operator<decltype(system), num::vec<double>, num::vec<double>>);

    num::vec<double> rhs(generator.rows(), 0.0);
    for (num::idx state = 0; state < rhs.size(); ++state) {
        rhs[state] = 1.0 + 0.1 * std::cos(static_cast<double>(state));
    }
    const num::gmres_options options{
        .tolerance = 1e-10, .max_iterations = 200, .restart = generator.rows()};
    num::vec<double> plain_solution(generator.rows(), 0.0);
    const auto plain = num::gmres(system, rhs, plain_solution, options);
    num::vec<double> multigrid_solution(generator.rows(), 0.0);
    const auto preconditioned =
        num::gmres(system, preconditioner, rhs, multigrid_solution, options);
    std::cout << "RDME GMRES iterations: " << plain.iterations << " -> "
              << preconditioned.iterations << " with sector multigrid\n";
    check::that(plain.converged, "plain.converged");
    check::that(preconditioned.converged, "preconditioned.converged");
    check::that(preconditioned.iterations <= plain.iterations, "preconditioned.iterations <= plain.iterations");

    for (num::idx state = 0; state < generator.rows(); ++state) {
        check::that(std::abs(multigrid_solution[state] - plain_solution[state]) < 1e-8, "std::abs(multigrid_solution[state] - plain_solution[state]) < 1e-8");
    }
}

void test_two_particle_annihilation_time_transform() {
    num::structured_grid_graph<1> grid({4}, {0.25});
    rdme::voxel_domain domain(std::move(grid), num::array<double>{1.0, 1.0});
    num::array<rdme::local_reaction> reactions{rdme::local_reaction{
        {-1, -1},
        [](num::view<const int> counts, num::view<const double> parameters, double volume, double) {
            return parameters[0] * static_cast<double>(counts[0] * counts[1]) / volume;
        }}};
    rdme::reaction_diffusion_model model(std::move(domain), std::move(reactions));
    rdme::restricted_generator generator(std::move(model), num::array<int>{1, 1},
                                         num::array<double>{2.0});
    check::that(generator.rows() == 25, "generator.rows() == 25"); // 1 + V + V + V^2

    constexpr double shift = 0.5;
    rdme::shifted_generator system(generator, shift);
    rdme::sector_diffusion_multigrid multigrid(generator.model().domain(), generator.state_space(),
                                               shift);
    const num::gmres_options options{
        .tolerance = 1e-12, .max_iterations = 100, .restart = generator.rows()};

    num::idx joint_sector = rdme::reaction_state_space::npos;
    for (num::idx sector = 0; sector < generator.state_space().n_sectors(); ++sector) {
        if (generator.state_space().sector_totals()[sector] == num::array<int>({1, 1})) {
            joint_sector = sector;
            break;
        }
    }
    check::that(joint_sector != rdme::reaction_state_space::npos, "joint_sector != rdme::reaction_state_space::npos");
    const num::idx joint_begin = generator.state_space().sector_offsets()[joint_sector];
    const num::idx joint_end = generator.state_space().sector_offsets()[joint_sector + 1];
    const num::idx vacuum = generator.state_space().find(rdme::configuration(8, 0));

    const auto transform = [&](std::size_t a_voxel, std::size_t b_voxel) {
        rdme::configuration initial(8, 0);
        initial[generator.state_space().layout().index(a_voxel, 0)] = 1;
        initial[generator.state_space().layout().index(b_voxel, 1)] = 1;
        num::vec<double> rhs(generator.rows(), 0.0);
        rhs[generator.state_space().find(initial)] = 1.0;
        num::vec<double> solution(generator.rows(), 0.0);
        const auto result = num::gmres(system, multigrid, rhs, solution, options);
        check::that(result.converged, "result.converged");
        double survival_transform = 0.0;
        double reaction_flux_transform = 0.0;
        for (num::idx state = joint_begin; state < joint_end; ++state) {
            survival_transform += solution[state];
            const auto &configuration = generator.state_space()[state];
            for (std::size_t voxel = 0; voxel < 4; ++voxel) {
                const auto &layout = generator.state_space().layout();
                reaction_flux_transform +=
                    2.0 *
                    static_cast<double>(configuration[layout.index(voxel, 0)] *
                                        configuration[layout.index(voxel, 1)]) /
                    generator.model().domain().voxel_volume(voxel) * solution[state];
            }
        }
        const double reaction_time_transform = shift * solution[vacuum];
        check::that(std::abs(reaction_time_transform - reaction_flux_transform) < 1e-10, "std::abs(reaction_time_transform - reaction_flux_transform) < 1e-10");
        check::that(std::abs(reaction_time_transform + shift * survival_transform - 1.0) < 1e-10, "std::abs(reaction_time_transform + shift * survival_transform - 1.0...");
        return reaction_time_transform;
    };

    const double colocated = transform(0, 0);
    const double separated = transform(0, 3);
    check::that(colocated > separated, "colocated > separated");
    check::that(separated > 0.0 && colocated < 1.0, "separated > 0.0 && colocated < 1.0");
}

int main() {
    test_structured_voxel_domain();
    test_explicit_graph_voxel_domain();
    test_exact_cme_lift();
    test_exact_cme_lift_supports_dynamic_states();
    test_dynamic_diffusion_sectors_scale_past_cme_limit();
    test_cached_matrix_free_diffusion_generator();
    test_reactions_couple_contiguous_population_sectors();
    test_population_conserving_recursive_multigrid();
    test_sector_multigrid_preconditions_reaction_diffusion_resolvent();
    test_two_particle_annihilation_time_transform();
    std::cout << "All RDME tests passed.\n";
    return check::report("rdme");
}
