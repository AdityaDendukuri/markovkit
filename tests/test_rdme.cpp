#include "check.hpp"
#include "linear/solvers/gmres.hpp"
#include "markovkit.hpp"
#include "structures/graph/graph.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace markovkit;

namespace {

real sum(const vec<real> &x) { return std::accumulate(x.begin(), x.end(), 0.0); }

// a birth at rate birth * volume and a death at rate death * count, in every voxel
array<local_reaction> birth_death(real birth, real death) {
    return {{{1}, [=](view<const int>, real volume) { return birth * volume; }},
            {{-1}, [=](view<const int> n, real) { return death * n[0]; }}};
}

void test_structured_voxel_domain() {
    voxel_domain domain(grid<2>({3, 2}, {0.5, 2.0}), {0.25, 1.0});
    check::that(domain.n_voxels() == 6 && domain.n_species() == 2, "domain size");
    check::that(std::abs(domain.volumes[0] - 1.0) < 1e-15, "cell volume");

    const auto neighbors = domain.graph.neighbors(0);
    check::that(neighbors.size() == 2, "corner voxel has two neighbors");
    check::that(std::abs(domain.jump_rate(0, 0, neighbors.begin()[0]) - 1.0) < 1e-15,
                "jump rate along the fine axis");
    check::that(std::abs(domain.jump_rate(0, 0, neighbors.begin()[1]) - 0.0625) < 1e-15,
                "jump rate along the coarse axis");
}

void test_explicit_graph_voxel_domain() {
    num::graph<double, num::idx> graph(2);
    graph.add_edge(0, 1, 3.0);
    voxel_domain domain(std::move(graph), {2.0, 4.0}, {0.5});
    check::that(std::abs(domain.jump_rate(0, 0, domain.graph.neighbors(0)[0]) - 0.75) < 1e-15,
                "jump rate out of the small voxel");
    check::that(std::abs(domain.jump_rate(0, 1, domain.graph.neighbors(1)[0]) - 0.375) < 1e-15,
                "jump rate out of the large voxel");
}

void test_reaction_diffusion_transitions() {
    // h = 0.5: transmissibility 2 and volume 0.5, hence 4 per molecule
    reaction_diffusion model{voxel_domain(grid<1>({2}, {0.5}), {1.0}),
                             {{{1}, [](view<const int>, real) { return 0.25; }}}};
    static_assert(markov_model<decltype(model)>);

    const auto jumps = model.transitions({3, 2});
    check::that(jumps.size() == 4, "two diffusion jumps and two births");
    const auto rate_to = [&](const state &y) {
        for (const auto &e : jumps) {
            if (e.to == y) {
                return e.rate;
            }
        }
        return 0.0;
    };
    check::that(std::abs(rate_to({2, 3}) - 12.0) < 1e-15, "three molecules hop right");
    check::that(std::abs(rate_to({4, 1}) - 8.0) < 1e-15, "two molecules hop left");
    check::that(std::abs(rate_to({4, 2}) - 0.25) < 1e-15, "birth in voxel 0");
    check::that(std::abs(rate_to({3, 3}) - 0.25) < 1e-15, "birth in voxel 1");

    // the RDME is a markov_model, so SSA runs on it directly; diffusion conserves molecules
    reaction_diffusion diffusion{voxel_domain(grid<1>({3}), {1.0}), {}};
    num::rng random(3);
    const auto path = gillespie(diffusion, {4, 0, 0}, 5.0, random);
    check::that(std::all_of(path.states.begin(), path.states.end(),
                            [](const state &x) { return x[0] + x[1] + x[2] == 4; }),
                "SSA on an RDME conserves a diffusing population");
}

void test_population_spaces() {
    check::that(population_sector(3, {4}).size() == 15, "15 ways to put 4 in 3 voxels");
    check::that(population_sector(3, {16}).size() == 153, "153 ways to put 16 in 3 voxels");
    check::that(population_sector(3, {64}).size() == 2145, "2145 ways to put 64 in 3 voxels");

    population_sector two_species(2, {1, 2});
    static_assert(state_space<population_sector>);
    check::that(two_species.size() == 6, "two-species sector size");
    for (idx i = 0; i < two_species.size(); ++i) {
        const state &x = two_species[i];
        check::that(x[0] + x[2] == 1 && x[1] + x[3] == 2, "sector totals");
        check::that(two_species.find(x) == i, "arithmetic rank reproduces enumeration");
    }
    check::that(!two_species.find({2, 0, 0, 2}), "a state with other totals is absent");

    population_box box(3, {2, 2});
    static_assert(state_space<population_box>);
    for (idx i = 0; i < box.size(); ++i) {
        const idx k = box.sector_of(i);
        check::that(box.find(box[i]) == i, "box rank reproduces enumeration");
        check::that(box.sector_begin(k) <= i && i < box.sector_end(k), "sector_of");
    }
    check::that(!box.find({3, 0, 0, 0, 0, 0}), "a state outside the box is absent");
}

void test_diffusion_generator() {
    population_sector space(3, {4});
    reaction_diffusion diffusion{voxel_domain(grid<1>({3}, {0.5}), {0.25}), {}};
    const generator Q(space, diffusion);
    static_assert(num::linear_operator<generator, vec<real>, vec<real>>);
    check::that(Q.rows() == 15, "one row per configuration");

    // h = 0.5 gives transmissibility 2 and volume 0.5: D T / V = 1 per molecule
    const idx source = *space.find({4, 0, 0});
    check::that(std::abs(Q.exit_rates()[source] - 4.0) < 1e-15, "four molecules can hop");

    vec<real> p(Q.rows(), 0.0), free(Q.rows(), 0.0), assembled(Q.rows(), 0.0);
    p[source] = 1.0;
    Q.apply(p, free);
    Q.matrix().apply(p, assembled);
    check::that(std::abs(free[source] + 4.0) < 1e-15, "diagonal");
    check::that(std::abs(sum(free)) < 1e-15, "diffusion conserves probability");
    for (idx i = 0; i < Q.rows(); ++i) {
        check::that(std::abs(free[i] - assembled[i]) < 1e-15, "matrix() matches apply()");
    }
}

void test_reactions_couple_neighboring_sectors() {
    reaction_diffusion model{voxel_domain(grid<1>({2}, {1.0}), {0.5}), birth_death(2.0, 0.25)};
    population_box box(2, {3});
    const generator Q(box, model, boundary::absorbing);
    check::that(Q.rows() == 10, "sector k holds k + 1 configurations of two voxels");

    bool diffusion = false, birth = false, death = false;
    for (idx j = 0; j < Q.rows(); ++j) {
        for (const auto &e : Q.jumps(j)) {
            const auto change = static_cast<long>(box.sector_of(e.to)) - long(box.sector_of(j));
            check::that(std::abs(change) <= 1, "reactions change a total by at most one");
            diffusion = diffusion || change == 0;
            birth = birth || change == 1;
            death = death || change == -1;
        }
    }
    check::that(diffusion && birth && death, "block tridiagonal across sectors");

    const idx empty = *box.find({0, 0});
    const idx full = *box.find({3, 0});
    check::that(std::abs(Q.exit_rates()[empty] - 4.0) < 1e-15, "two births from empty");
    // 1.5 diffusion + 0.75 death + 4.0 birth leaking out of the box
    check::that(std::abs(Q.exit_rates()[full] - 6.25) < 1e-15, "absorbing exit rate");

    vec<real> p(Q.rows(), 0.0), free(Q.rows(), 0.0), assembled(Q.rows(), 0.0);
    p[full] = 1.0;
    Q.apply(p, free);
    Q.matrix().apply(p, assembled);
    check::that(std::abs(sum(free) + 4.0) < 1e-15, "only the leaking births lose probability");
    for (idx i = 0; i < Q.rows(); ++i) {
        check::that(std::abs(free[i] - assembled[i]) < 1e-15, "matrix() matches apply()");
    }

    const generator R(box, model, boundary::reflecting);
    check::that(std::abs(R.exit_rates()[full] - 2.25) < 1e-15, "reflecting drops the leak");
    check::that(std::abs(R.leak_rates()[full] - 4.0) < 1e-15, "the leak is still recorded");
}

void test_population_multigrid() {
    voxel_domain domain(grid<1>({8}, {1.0}), {1.0});
    population_multigrid multigrid(domain, {2}, 1.0);
    static_assert(num::linear_operator<decltype(multigrid), vec<real>, vec<real>>);
    check::that(multigrid.rows() == 36, "fine sector size");
    check::that(multigrid.transfer().coarse_size() == 10, "coarse sector size");
    check::that(multigrid.n_levels() == 3, "levels");
    check::that(multigrid.grid_complexity() >= 1.0 && multigrid.operator_complexity() >= 1.0,
                "complexities");

    // R P = I: aggregation exactly reverses the multinomial lift
    vec<real> coarse(10, 0.0), fine(36, 0.0), restricted(10, 0.0);
    for (idx i = 0; i < coarse.size(); ++i) {
        coarse[i] = std::sin(real(i));
    }
    multigrid.transfer().prolong(coarse, fine);
    multigrid.transfer().restrict(fine, restricted);
    for (idx i = 0; i < coarse.size(); ++i) {
        check::that(std::abs(restricted[i] - coarse[i]) < 1e-14, "R P = I");
    }
    check::that(std::abs(sum(fine) - sum(coarse)) < 1e-14, "prolongation conserves mass");

    // rediscretizing with doubled diffusivity is the Petrov-Galerkin operator R A P
    vec<real> fine_image(36, 0.0), projected(10, 0.0), coarse_image(10, 0.0);
    multigrid.apply_system(fine, fine_image);
    multigrid.transfer().restrict(fine_image, projected);
    multigrid.coarse_level()->apply_system(coarse, coarse_image);
    for (idx i = 0; i < coarse.size(); ++i) {
        check::that(std::abs(projected[i] - coarse_image[i]) < 1e-12, "R A P = coarse A");
    }

    const auto residual_norm = [&](const population_multigrid<1> &cycle, const vec<real> &b) {
        vec<real> x(b.size(), 0.0), image(b.size(), 0.0);
        cycle.apply(b, x);
        cycle.apply_system(x, image);
        real squared = 0.0;
        for (idx i = 0; i < b.size(); ++i) {
            squared += (b[i] - image[i]) * (b[i] - image[i]);
        }
        return squared;
    };
    vec<real> b(36, 0.0);
    real initial = 0.0;
    for (idx i = 0; i < b.size(); ++i) {
        b[i] = 0.25 + std::sin(real(i));
        initial += b[i] * b[i];
    }
    const real jacobi = residual_norm(multigrid, b);
    check::that(jacobi < initial, "a Jacobi V-cycle reduces the residual");

    // aggregate blocks couple the states that differ only inside one coarse voxel
    population_multigrid block(domain, {2}, 1.0, {.smoother = relaxation::aggregate_block});
    check::that(residual_norm(block, b) < jacobi, "aggregate blocks beat point Jacobi");

    population_box box(8, {2});
    sector_multigrid sectors(domain, box, 1.0);
    static_assert(num::linear_operator<decltype(sectors), vec<real>, vec<real>>);
    check::that(sectors.rows() == 45 && sectors.n_sectors() == 3, "one cycle per sector");
    vec<real> ones(45, 1.0), correction(45, 0.0);
    sectors.apply(ones, correction);
    check::that(std::all_of(correction.begin(), correction.end(),
                            [](real v) { return std::isfinite(v); }),
                "sector multigrid is finite");
}

void test_multigrid_preconditions_resolvent() {
    reaction_diffusion model{voxel_domain(grid<1>({4}, {1.0}), {4.0}), birth_death(0.05, 0.02)};
    population_box box(4, {3});
    const generator Q(box, model, boundary::absorbing);
    constexpr real shift = 0.5;
    const shifted_generator A(Q, shift);
    const sector_multigrid preconditioner(model.domain, box, shift);
    static_assert(num::linear_operator<shifted_generator, vec<real>, vec<real>>);

    vec<real> b(Q.rows(), 0.0);
    for (idx i = 0; i < b.size(); ++i) {
        b[i] = 1.0 + 0.1 * std::cos(real(i));
    }
    const num::gmres_options options{.tolerance = 1e-10, .max_iterations = 200, .restart = Q.rows()};
    vec<real> x_plain(Q.rows(), 0.0), x_mg(Q.rows(), 0.0);
    const auto plain = num::gmres(A, b, x_plain, options);
    const auto mg = num::gmres(A, preconditioner, b, x_mg, options);
    std::cout << "RDME GMRES iterations: " << plain.iterations << " -> " << mg.iterations
              << " with sector multigrid\n";
    check::that(plain.converged && mg.converged, "both solves converge");
    check::that(mg.iterations <= plain.iterations, "multigrid does not slow GMRES down");
    for (idx i = 0; i < Q.rows(); ++i) {
        check::that(std::abs(x_mg[i] - x_plain[i]) < 1e-8, "both solves agree");
    }
}

void test_two_particle_annihilation_transform() {
    // A + B -> nothing at rate 2 a b / V
    reaction_diffusion model{
        voxel_domain(grid<1>({4}, {0.25}), {1.0, 1.0}),
        {{{-1, -1}, [](view<const int> n, real volume) { return 2.0 * n[0] * n[1] / volume; }}}};
    population_box box(4, {1, 1});
    const generator Q(box, model);
    check::that(Q.rows() == 25, "1 + V + V + V^2 states");

    constexpr real shift = 0.5;
    const shifted_generator A(Q, shift);
    const sector_multigrid preconditioner(model.domain, box, shift);
    const num::gmres_options options{.tolerance = 1e-12, .max_iterations = 100, .restart = Q.rows()};
    const idx joint = *box.find_sector({1, 1});
    const idx vacuum = *box.find(state(8, 0));

    // E[exp(-s T)] for the annihilation time T, two ways
    const auto transform = [&](idx a_voxel, idx b_voxel) {
        state initial(8, 0);
        initial[model.index(a_voxel, 0)] = 1;
        initial[model.index(b_voxel, 1)] = 1;
        vec<real> b(Q.rows(), 0.0), x(Q.rows(), 0.0);
        b[*box.find(initial)] = 1.0;
        check::that(num::gmres(A, preconditioner, b, x, options).converged, "resolvent solve");
        real survival = 0.0, flux = 0.0;
        for (idx i = box.sector_begin(joint); i < box.sector_end(joint); ++i) {
            survival += x[i];
            for (idx v = 0; v < 4; ++v) {
                flux += 2.0 * box[i][model.index(v, 0)] * box[i][model.index(v, 1)] /
                        model.domain.volumes[v] * x[i];
            }
        }
        const real absorbed = shift * x[vacuum];
        check::that(std::abs(absorbed - flux) < 1e-10, "absorption equals reaction flux");
        check::that(std::abs(absorbed + shift * survival - 1.0) < 1e-10,
                    "E[exp(-sT)] + s S_hat(s) = 1");
        return absorbed;
    };
    const real colocated = transform(0, 0);
    const real separated = transform(0, 3);
    check::that(colocated > separated, "colocated particles react sooner");
    check::that(separated > 0.0 && colocated < 1.0, "transforms lie in (0, 1)");
}

} // namespace

int main() {
    test_structured_voxel_domain();
    test_explicit_graph_voxel_domain();
    test_reaction_diffusion_transitions();
    test_population_spaces();
    test_diffusion_generator();
    test_reactions_couple_neighboring_sectors();
    test_population_multigrid();
    test_multigrid_preconditions_resolvent();
    test_two_particle_annihilation_transform();
    return check::report("rdme");
}
