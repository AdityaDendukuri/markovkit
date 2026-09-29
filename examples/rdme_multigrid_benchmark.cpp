#include "core/types.hpp"
#include "linear/solvers/ilu.hpp"
#include "linear/solvers/math_gmres.hpp"
#include "linear/solvers/preconditioner.hpp"
#include "rdme/rdme.hpp"
#include "structures/graph/structured_grid.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

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
    num::structured_grid_graph<D> grid(dimensions, spacing);
    rdme::voxel_domain domain(std::move(grid), num::array<double>{problem.diffusivity});

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

    const auto generator_begin = clock::now();
    rdme::restricted_generator generator(
        std::move(model), num::array<int>{problem.max_population},
        num::array<double>{0.05 * problem.reaction_scale, 0.02 * problem.reaction_scale}, 0.0,
        rdme::boundary_mode::absorbing);
    const auto generator_end = clock::now();
    const double generator_setup_ms = milliseconds(generator_begin, generator_end);
    const num::idx voxels = generator.model().domain().n_voxels();

    rdme::shifted_generator system(generator, problem.shift);
    num::vec rhs(generator.rows(), 0.0);
    for (num::idx state = 0; state < rhs.size(); ++state) {
        rhs[state] = 1.0 + 0.1 * std::cos(static_cast<double>(state));
    }
    const num::gmres_options options{.tolerance = 1e-8,
                                     .max_iterations = 500,
                                     .restart = std::min<num::idx>(50, generator.rows())};

    num::vec plain_solution(generator.rows(), 0.0);
    const auto plain_begin = clock::now();
    const auto plain = num::gmres(system, rhs, plain_solution, options);
    const auto plain_end = clock::now();
    const double plain_solve_ms = milliseconds(plain_begin, plain_end);

    const auto jacobi_begin = clock::now();
    num::vec inverse_diagonal(generator.rows(), 0.0);
    for (num::idx state = 0; state < generator.rows(); ++state) {
        inverse_diagonal[state] = 1.0 / (problem.shift + generator.exit_rates()[state]);
    }
    num::jacobi_preconditioner jacobi(std::move(inverse_diagonal));
    const auto jacobi_end = clock::now();
    num::vec jacobi_solution(generator.rows(), 0.0);
    const auto jacobi_solve_begin = clock::now();
    const auto jacobi_result = num::gmres(system, jacobi, rhs, jacobi_solution, options);
    const auto jacobi_solve_end = clock::now();
    const double jacobi_setup_ms = milliseconds(jacobi_begin, jacobi_end);
    const double jacobi_solve_ms = milliseconds(jacobi_solve_begin, jacobi_solve_end);

    const auto ilu_begin = clock::now();
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
    const auto ilu_end = clock::now();
    num::vec ilu_solution(generator.rows(), 0.0);
    const auto ilu_solve_begin = clock::now();
    const auto ilu_result = num::gmres(system, ilu, rhs, ilu_solution, options);
    const auto ilu_solve_end = clock::now();
    const double ilu_setup_ms = milliseconds(ilu_begin, ilu_end);
    const double ilu_solve_ms = milliseconds(ilu_solve_begin, ilu_solve_end);

    const auto multigrid_begin = clock::now();
    rdme::sector_diffusion_multigrid multigrid(generator.model().domain(), generator.state_space(),
                                               problem.shift);
    const auto multigrid_end = clock::now();
    num::vec multigrid_solution(generator.rows(), 0.0);
    const auto multigrid_solve_begin = clock::now();
    const auto multigrid_result = num::gmres(system, multigrid, rhs, multigrid_solution, options);
    const auto multigrid_solve_end = clock::now();
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
