#include "linear/solvers/ilu.hpp"
#include "linear/solvers/math_gmres.hpp"
#include "linear/solvers/preconditioner.hpp"
#include "rdme/rdme.hpp"
#include "structures/graph/structured_grid.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Case {
    int dimension;
    num::idx cells_per_axis;
    int max_population;
    double diffusivity;
    double reaction_scale;
    double shift;

    bool operator==(const Case &) const = default;
};

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double relative_difference(const num::vec &solution, const num::vec &reference) {
    double difference_squared = 0.0;
    double reference_squared = 0.0;
    for (num::idx state = 0; state < solution.size(); ++state) {
        const double difference = solution[state] - reference[state];
        difference_squared += difference * difference;
        reference_squared += reference[state] * reference[state];
    }
    return std::sqrt(difference_squared / std::max(reference_squared, 1e-300));
}

void emit_row(const Case &problem, num::idx voxels, num::idx states, std::size_t levels,
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
void run_case(const Case &problem) {
    std::array<num::idx, D> dimensions{};
    dimensions.fill(problem.cells_per_axis);
    std::array<double, D> spacing{};
    spacing.fill(1.0 / static_cast<double>(problem.cells_per_axis));
    num::structured_grid_graph<D> grid(dimensions, spacing);
    rdme::VoxelDomain domain(std::move(grid), std::vector<double>{problem.diffusivity});

    num::array<rdme::LocalReaction> reactions{
        rdme::LocalReaction{{1},
                            [](std::span<const int>, std::span<const double> parameters,
                               double volume, double) { return parameters[0] * volume; }},
        rdme::LocalReaction{{-1},
                            [](std::span<const int> counts, std::span<const double> parameters,
                               double,
                               double) { return parameters[1] * static_cast<double>(counts[0]); }}};
    rdme::ReactionDiffusionModel model(std::move(domain), std::move(reactions));

    const auto generator_begin = Clock::now();
    rdme::RestrictedGenerator generator(
        std::move(model), num::array<int>{problem.max_population},
        num::array<double>{0.05 * problem.reaction_scale, 0.02 * problem.reaction_scale}, 0.0,
        rdme::BoundaryMode::absorbing);
    const auto generator_end = Clock::now();
    const double generator_setup_ms = milliseconds(generator_begin, generator_end);
    const num::idx voxels = generator.model().domain().n_voxels();

    rdme::ShiftedGenerator system(generator, problem.shift);
    num::vec rhs(generator.rows(), 0.0);
    for (num::idx state = 0; state < rhs.size(); ++state) {
        rhs[state] = 1.0 + 0.1 * std::cos(static_cast<double>(state));
    }
    const num::gmres_options options{.tolerance = 1e-8,
                                     .max_iterations = 500,
                                     .restart = std::min<num::idx>(50, generator.rows())};

    num::vec plain_solution(generator.rows(), 0.0);
    const auto plain_begin = Clock::now();
    const auto plain = num::gmres(system, rhs, plain_solution, options);
    const auto plain_end = Clock::now();
    const double plain_solve_ms = milliseconds(plain_begin, plain_end);

    const auto jacobi_begin = Clock::now();
    num::vec inverse_diagonal(generator.rows(), 0.0);
    for (num::idx state = 0; state < generator.rows(); ++state) {
        inverse_diagonal[state] = 1.0 / (problem.shift + generator.exit_rates()[state]);
    }
    num::jacobi_preconditioner jacobi(std::move(inverse_diagonal));
    const auto jacobi_end = Clock::now();
    num::vec jacobi_solution(generator.rows(), 0.0);
    const auto jacobi_solve_begin = Clock::now();
    const auto jacobi_result = num::gmres(system, jacobi, rhs, jacobi_solution, options);
    const auto jacobi_solve_end = Clock::now();
    const double jacobi_setup_ms = milliseconds(jacobi_begin, jacobi_end);
    const double jacobi_solve_ms = milliseconds(jacobi_solve_begin, jacobi_solve_end);

    const auto ilu_begin = Clock::now();
    num::array<num::idx> matrix_rows;
    num::array<num::idx> matrix_columns;
    num::array<double> matrix_values;
    matrix_rows.reserve(generator.transitions().size() + generator.rows());
    matrix_columns.reserve(matrix_rows.capacity());
    matrix_values.reserve(matrix_rows.capacity());
    for (num::idx source = 0; source < generator.rows(); ++source) {
        matrix_rows.push_back(source);
        matrix_columns.push_back(source);
        matrix_values.push_back(problem.shift + generator.exit_rates()[source]);
        for (num::idx entry = generator.source_offsets()[source];
             entry < generator.source_offsets()[source + 1]; ++entry) {
            matrix_rows.push_back(generator.transitions()[entry].destination);
            matrix_columns.push_back(source);
            matrix_values.push_back(-generator.transitions()[entry].rate);
        }
    }
    const num::spmat sparse_system = num::spmat::from_triplets(
        generator.rows(), generator.cols(), matrix_rows, matrix_columns, matrix_values);
    const num::ilu0_preconditioner ilu(sparse_system);
    const auto ilu_end = Clock::now();
    num::vec ilu_solution(generator.rows(), 0.0);
    const auto ilu_solve_begin = Clock::now();
    const auto ilu_result = num::gmres(system, ilu, rhs, ilu_solution, options);
    const auto ilu_solve_end = Clock::now();
    const double ilu_setup_ms = milliseconds(ilu_begin, ilu_end);
    const double ilu_solve_ms = milliseconds(ilu_solve_begin, ilu_solve_end);

    const auto multigrid_begin = Clock::now();
    rdme::SectorDiffusionMultigrid multigrid(generator.model().domain(), generator.state_space(),
                                             problem.shift);
    const auto multigrid_end = Clock::now();
    num::vec multigrid_solution(generator.rows(), 0.0);
    const auto multigrid_solve_begin = Clock::now();
    const auto multigrid_result = num::gmres(system, multigrid, rhs, multigrid_solution, options);
    const auto multigrid_solve_end = Clock::now();
    const double multigrid_setup_ms = milliseconds(multigrid_begin, multigrid_end);
    const double multigrid_solve_ms = milliseconds(multigrid_solve_begin, multigrid_solve_end);

    const num::vec *reference = nullptr;
    if (plain.converged)
        reference = &plain_solution;
    else if (multigrid_result.converged)
        reference = &multigrid_solution;
    else if (ilu_result.converged)
        reference = &ilu_solution;
    else if (jacobi_result.converged)
        reference = &jacobi_solution;
    const auto difference = [&](const num::vec &solution, bool converged) {
        return reference && converged ? relative_difference(solution, *reference)
                                      : std::numeric_limits<double>::quiet_NaN();
    };

    emit_row(problem, voxels, generator.rows(), 0, "plain", generator_setup_ms, 0.0, plain_solve_ms,
             plain, difference(plain_solution, plain.converged));
    emit_row(problem, voxels, generator.rows(), 0, "jacobi", generator_setup_ms, jacobi_setup_ms,
             jacobi_solve_ms, jacobi_result, difference(jacobi_solution, jacobi_result.converged));
    emit_row(problem, voxels, generator.rows(), 0, "ilu0", generator_setup_ms, ilu_setup_ms,
             ilu_solve_ms, ilu_result, difference(ilu_solution, ilu_result.converged));
    emit_row(problem, voxels, generator.rows(), multigrid.max_hierarchy_levels(), "multigrid",
             generator_setup_ms, multigrid_setup_ms, multigrid_solve_ms, multigrid_result,
             difference(multigrid_solution, multigrid_result.converged));
}

void dispatch(const Case &problem) {
    if (problem.dimension == 1)
        return run_case<1>(problem);
    if (problem.dimension == 2)
        return run_case<2>(problem);
    if (problem.dimension == 3)
        return run_case<3>(problem);
    throw std::invalid_argument("RDME benchmark dimension must be 1, 2, or 3");
}

std::vector<Case> quick_cases() {
    return {{1, 4, 3, 4.0, 1.0, 0.5},
            {1, 8, 3, 4.0, 1.0, 0.5},
            {1, 16, 3, 4.0, 1.0, 0.5},
            {2, 4, 2, 4.0, 1.0, 0.5},
            {3, 2, 2, 4.0, 1.0, 0.5}};
}

std::vector<Case> paper_cases() {
    std::vector<Case> cases;
    const auto add = [&](int dimension, num::idx cells, int population, double diffusion,
                         double reaction, double shift) {
        const Case candidate{dimension, cells, population, diffusion, reaction, shift};
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
         {std::array<int, 3>{1, 16, 3}, std::array<int, 3>{2, 8, 2}, std::array<int, 3>{3, 4, 2}}) {
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
    for (const Case &problem : quick ? quick_cases() : paper_cases())
        dispatch(problem);
    return 0;
}
