// plain and preconditioned GMRES for RDME resolvent solves; tidy CSV on stdout
#include "core/types.hpp"
#include "linear/solvers/ilu.hpp"
#include "linear/solvers/gmres.hpp"
#include "linear/solvers/preconditioner.hpp"
#include "markovkit.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

using namespace markovkit;

namespace {

using clock = std::chrono::steady_clock;

struct benchmark_case {
    int dimension;
    num::idx cells_per_axis;
    int max_population;
    double diffusivity;
    double reaction_scale;
    double shift;

    bool operator==(const benchmark_case &) const = default;
};

double milliseconds(clock::time_point begin, clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double relative_difference(const num::vec<double> &solution, const num::vec<double> &reference) {
    double difference_squared = 0.0;
    double reference_squared = 0.0;
    for (num::idx state = 0; state < solution.size(); ++state) {
        const double difference = solution[state] - reference[state];
        difference_squared += difference * difference;
        reference_squared += reference[state] * reference[state];
    }
    return std::sqrt(difference_squared / std::max(reference_squared, 1e-300));
}

void emit_row(const benchmark_case &problem, num::idx voxels, num::idx states, std::size_t levels,
              std::string_view method, double generator_setup_ms, double preconditioner_setup_ms,
              double solve_ms, const num::solver_result &result, double solution_difference) {
    std::cout << problem.dimension << ',' << problem.cells_per_axis << ',' << voxels << ','
              << problem.max_population << ',' << states << ',' << problem.diffusivity << ','
              << problem.reaction_scale << ',' << problem.shift << ',' << levels << ',' << method
              << ',' << generator_setup_ms << ',' << preconditioner_setup_ms << ',' << solve_ms
              << ',' << result.iterations << ',' << result.residual << ',' << result.converged
              << ',' << solution_difference << '\n';
}

template <std::size_t D>
void run_case(const benchmark_case &problem) {
    num::static_array<num::idx, D> dimensions{};
    dimensions.fill(problem.cells_per_axis);
    num::static_array<double, D> spacing{};
    spacing.fill(1.0 / static_cast<double>(problem.cells_per_axis));
    const double birth = 0.05 * problem.reaction_scale;
    const double death = 0.02 * problem.reaction_scale;
    const reaction_diffusion model{
        voxel_domain(grid<D>(dimensions, spacing), {problem.diffusivity}),
        {{{1}, [=](view<const int>, real volume) { return birth * volume; }},
         {{-1}, [=](view<const int> n, real) { return death * n[0]; }}}};

    const auto generator_begin = clock::now();
    const population_box box(model.n_voxels(), {problem.max_population});
    const generator Q(box, model, boundary::absorbing);
    const double generator_setup_ms = milliseconds(generator_begin, clock::now());

    const shifted_generator system(Q, problem.shift);
    vec<real> rhs(Q.rows(), 0.0);
    for (idx i = 0; i < rhs.size(); ++i) {
        rhs[i] = 1.0 + 0.1 * std::cos(static_cast<double>(i));
    }
    const num::gmres_options options{
        .tolerance = 1e-8, .max_iterations = 500, .restart = std::min<idx>(50, Q.rows())};

    // set up a preconditioner, then solve with it, timing both
    struct run {
        num::solver_result result;
        vec<real> solution;
        double setup_ms, solve_ms;
    };
    const auto solve = [&](auto make_preconditioner) {
        const auto setup_begin = clock::now();
        const auto preconditioner = make_preconditioner();
        const auto solve_begin = clock::now();
        vec<real> solution(Q.rows(), 0.0);
        const auto result = num::gmres(system, preconditioner, rhs, solution, options);
        return run{result, std::move(solution), milliseconds(setup_begin, solve_begin),
                   milliseconds(solve_begin, clock::now())};
    };

    vec<real> plain_solution(Q.rows(), 0.0);
    const auto plain_begin = clock::now();
    const auto plain_result = num::gmres(system, rhs, plain_solution, options);
    const run plain{plain_result, std::move(plain_solution), 0.0,
                    milliseconds(plain_begin, clock::now())};

    const run jacobi = solve([&] {
        vec<real> inverse_diagonal(Q.rows(), 0.0);
        for (idx i = 0; i < Q.rows(); ++i) {
            inverse_diagonal[i] = 1.0 / (problem.shift + Q.exit_rates()[i]);
        }
        return num::jacobi_preconditioner(std::move(inverse_diagonal));
    });
    const run ilu = solve([&] { return num::ilu0_preconditioner(system.matrix()); });
    std::size_t levels = 0;
    const run multigrid = solve([&] {
        sector_multigrid<D> preconditioner(model.domain, box, problem.shift);
        levels = preconditioner.max_levels();
        return preconditioner;
    });

    const run *reference = nullptr;
    for (const run *candidate : {&plain, &multigrid, &ilu, &jacobi}) {
        if (!reference && candidate->result.converged) {
            reference = candidate;
        }
    }
    const auto emit = [&](std::string_view method, const run &r, std::size_t run_levels) {
        const double difference = reference && r.result.converged
                                      ? relative_difference(r.solution, reference->solution)
                                      : std::numeric_limits<double>::quiet_NaN();
        emit_row(problem, model.n_voxels(), Q.rows(), run_levels, method, generator_setup_ms,
                 r.setup_ms, r.solve_ms, r.result, difference);
    };
    emit("plain", plain, 0);
    emit("jacobi", jacobi, 0);
    emit("ilu0", ilu, 0);
    emit("multigrid", multigrid, levels);
}

void dispatch(const benchmark_case &problem) {
    if (problem.dimension == 1)
        return run_case<1>(problem);
    if (problem.dimension == 2)
        return run_case<2>(problem);
    if (problem.dimension == 3)
        return run_case<3>(problem);
    throw std::invalid_argument("RDME benchmark dimension must be 1, 2, or 3");
}

num::array<benchmark_case> quick_cases() {
    return {{1, 4, 3, 4.0, 1.0, 0.5},
            {1, 8, 3, 4.0, 1.0, 0.5},
            {1, 16, 3, 4.0, 1.0, 0.5},
            {2, 4, 2, 4.0, 1.0, 0.5},
            {3, 2, 2, 4.0, 1.0, 0.5}};
}

num::array<benchmark_case> paper_cases() {
    num::array<benchmark_case> cases;
    const auto add = [&](int dimension, num::idx cells, int population, double diffusion,
                         double reaction, double shift) {
        const benchmark_case candidate{dimension, cells, population, diffusion, reaction, shift};
        if (std::find(cases.begin(), cases.end(), candidate) == cases.end()) {
            cases.push_back(candidate);
        }
    };

    for (num::idx cells : {num::idx{4}, num::idx{8}, num::idx{16}, num::idx{32}})
        for (int population : {1, 2, 3})
            add(1, cells, population, 4.0, 1.0, 0.5);
    for (num::idx cells : {num::idx{2}, num::idx{4}})
        for (int population : {1, 2, 3})
            add(2, cells, population, 4.0, 1.0, 0.5);
    for (int population : {1, 2})
        add(2, 8, population, 4.0, 1.0, 0.5);
    for (int population : {1, 2, 3})
        add(3, 2, population, 4.0, 1.0, 0.5);
    for (int population : {1, 2})
        add(3, 4, population, 4.0, 1.0, 0.5);

    for (const auto [dimension, cells, population] :
         {num::static_array<int, 3>{1, 16, 3}, num::static_array<int, 3>{2, 8, 2},
          num::static_array<int, 3>{3, 4, 2}}) {
        for (double diffusion : {0.1, 1.0, 10.0, 100.0})
            add(dimension, cells, population, diffusion, 1.0, 0.5);
        for (double reaction : {0.01, 0.1, 1.0, 10.0})
            add(dimension, cells, population, 4.0, reaction, 0.5);
        for (double shift : {0.01, 0.1, 1.0, 10.0})
            add(dimension, cells, population, 4.0, 1.0, shift);
    }
    return cases;
}

} // namespace

int main(int argc, char **argv) {
    bool quick = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--quick") {
            quick = true;
        } else if (value == "--help") {
            std::cerr << "usage: rdme_multigrid_benchmark [--quick]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << value << '\n';
            return 2;
        }
    }

    std::cout << std::setprecision(10);
    std::cout << "dimension,cells_per_axis,voxels,max_population,states,diffusivity,"
                 "reaction_scale,shift,levels,method,generator_setup_ms,"
                 "preconditioner_setup_ms,solve_ms,iterations,residual,converged,"
                 "relative_solution_difference\n";
    for (const benchmark_case &problem : quick ? quick_cases() : paper_cases())
        dispatch(problem);
    return 0;
}
