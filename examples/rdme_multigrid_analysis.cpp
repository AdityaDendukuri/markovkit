// smoothing and two-grid analysis of population multigrid on fixed-population sectors
#include "core/types.hpp"
#include "plot/plot.hpp"
#include "markovkit.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace markovkit;

namespace {

struct benchmark_case {
    int dimension;
    num::idx cells_per_axis;
    int population;
    double shift;
};

struct result {
    benchmark_case problem;
    num::idx voxels;
    num::idx states;
    std::size_t levels;
    double grid_complexity;
    double operator_complexity;
    double smoothing_factor_one;
    double smoothing_factor_two;
    double block_smoothing_factor;
    double two_grid_factor;
    double block_two_grid_factor;
    double vcycle_factor;
    double block_vcycle_factor;
    double block_wcycle_factor;
    double wcycle_factor;
    double variable_vcycle_factor;
};

num::array<result> results;

double weighted_norm(num::view<const double> x, num::view<const double> stationary) {
    double squared = 0.0;
    for (std::size_t state = 0; state < x.size(); ++state)
        squared += x[state] * x[state] / stationary[state];
    return std::sqrt(squared);
}

void normalize(num::vec<double> &x, num::view<const double> stationary) {
    const double norm = weighted_norm(x, stationary);
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::runtime_error("RDME multigrid analysis encountered a zero error vector");
    for (double &value : x)
        value /= norm;
}

// the multinomial stationary distribution of diffusion in a sector of equal voxels
vec<real> stationary_distribution(const population_sector &space) {
    const idx S = space.n_species();
    vec<real> stationary(space.size(), 0.0);
    for (idx i = 0; i < space.size(); ++i) {
        double log_probability = 0.0;
        for (idx species = 0; species < S; ++species) {
            const int total = space.totals()[species];
            log_probability += std::lgamma(total + 1.0) -
                               (total * std::log(static_cast<double>(space.n_voxels())));
            for (idx voxel = 0; voxel < space.n_voxels(); ++voxel) {
                log_probability -= std::lgamma(space[i][(voxel * S) + species] + 1.0);
            }
        }
        stationary[i] = std::exp(log_probability);
    }
    double mass = 0.0;
    for (double value : stationary)
        mass += value;
    for (double &value : stationary)
        value /= mass;
    return stationary;
}

template <typename Transfer>
void remove_coarse_component(const Transfer &transfer, num::vec<double> &fine, num::vec<double> &coarse,
                             num::vec<double> &lifted) {
    transfer.restrict(fine, coarse);
    transfer.prolong(coarse, lifted);
    for (num::idx state = 0; state < fine.size(); ++state)
        fine[state] -= lifted[state];
}

template <typename Apply, typename Project>
double estimate_asymptotic_factor(num::idx size, num::view<const double> stationary, Apply apply,
                                  Project project, std::uint64_t seed, int iterations = 50,
                                  int probes = 3) {
    std::mt19937_64 engine(seed);
    std::normal_distribution<double> normal;
    double estimate = 0.0;
    for (int probe = 0; probe < probes; ++probe) {
        num::vec<double> error(size, 0.0);
        num::vec<double> next(size, 0.0);
        for (num::idx state = 0; state < size; ++state)
            error[state] = std::sqrt(stationary[state]) * normal(engine);
        project(error);
        normalize(error, stationary);

        double tail_log_sum = 0.0;
        int tail_count = 0;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            apply(error, next);
            project(next);
            const double ratio = weighted_norm(next, stationary);
            if (!(ratio > 0.0) || !std::isfinite(ratio)) {
                tail_log_sum = -std::numeric_limits<double>::infinity();
                tail_count = 1;
                break;
            }
            if (iteration >= iterations - 10) {
                tail_log_sum += std::log(ratio);
                ++tail_count;
            }
            for (num::idx state = 0; state < size; ++state)
                error[state] = next[state] / ratio;
        }
        const double probe_estimate =
            tail_count == 0 ? 0.0 : std::exp(tail_log_sum / static_cast<double>(tail_count));
        estimate = std::max(estimate, probe_estimate);
    }
    return estimate;
}

template <std::size_t D>
void run_case(const benchmark_case &problem) {
    num::static_array<num::idx, D> dimensions{};
    dimensions.fill(problem.cells_per_axis);
    num::static_array<double, D> spacing{};
    spacing.fill(1.0 / static_cast<double>(problem.cells_per_axis));
    const voxel_domain domain(grid<D>(dimensions, spacing), {1.0});
    const array<int> totals{problem.population};
    const real s = problem.shift;
    using cycle = population_multigrid<D>;
    const auto block = relaxation::aggregate_block;

    const cycle multigrid(domain, totals, s);
    const cycle block_multigrid(domain, totals, s, {.smoother = block});
    const cycle block_wcycle(domain, totals, s, {.coarse_cycles = 2, .smoother = block});
    const cycle wcycle(domain, totals, s, {.coarse_cycles = 2});
    const cycle variable_vcycle(domain, totals, s, {.sweep_growth = 4});
    const cycle two_grid(domain, totals, s, {.coarsest_size = 1, .max_levels = 2});
    const cycle block_two_grid(domain, totals, s,
                               {.coarsest_size = 1, .max_levels = 2, .smoother = block});
    if (!two_grid.coarse_level())
        throw std::logic_error("RDME two-grid analysis case has no coarse level");

    const generator &Q = multigrid.fine_generator();
    const auto &transfer = two_grid.transfer();
    const vec<real> stationary = stationary_distribution(multigrid.space());
    num::vec<double> system_image(Q.rows(), 0.0);
    num::vec<double> work(Q.rows(), 0.0);
    num::vec<double> coarse(transfer.coarse_size(), 0.0);
    num::vec<double> lifted(Q.rows(), 0.0);
    num::vec<double> coarse_residual(transfer.coarse_size(), 0.0);
    num::vec<double> coarse_correction(transfer.coarse_size(), 0.0);
    num::vec<double> zero(Q.rows(), 0.0);
    constexpr double omega = 2.0 / 3.0;

    const auto smooth = [&](const num::vec<double> &input, num::vec<double> &output, int sweeps) {
        output = input;
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            multigrid.apply_system(output, system_image);
            for (num::idx state = 0; state < output.size(); ++state) {
                const double diagonal = problem.shift + Q.exit_rates()[state];
                work[state] = output[state] - omega * system_image[state] / diagonal;
            }
            std::swap(output, work);
        }
    };
    const auto high_pass = [&](num::vec<double> &error) {
        remove_coarse_component(transfer, error, coarse, lifted);
    };
    const auto no_projection = [](num::vec<double> &) {};

    const double smoothing_one = estimate_asymptotic_factor(
        Q.rows(), stationary,
        [&](const num::vec<double> &input, num::vec<double> &output) { smooth(input, output, 1); }, high_pass,
        0x51a7ULL + problem.cells_per_axis * 31 + problem.population);
    const double smoothing_two = estimate_asymptotic_factor(
        Q.rows(), stationary,
        [&](const num::vec<double> &input, num::vec<double> &output) { smooth(input, output, 2); }, high_pass,
        0x72b9ULL + problem.cells_per_axis * 31 + problem.population);
    const double block_smoothing = estimate_asymptotic_factor(
        Q.rows(), stationary,
        [&](const num::vec<double> &input, num::vec<double> &output) {
            output = input;
            block_multigrid.relax(zero, output, 1);
        },
        high_pass, 0x83baULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_two_grid = [&](const num::vec<double> &input, num::vec<double> &output) {
        smooth(input, output, 2);
        two_grid.apply_system(output, system_image);
        transfer.restrict(system_image, coarse_residual);
        two_grid.coarse_level()->cycle(coarse_residual, coarse_correction);
        transfer.prolong(coarse_correction, lifted);
        for (num::idx state = 0; state < output.size(); ++state)
            output[state] -= lifted[state];
        work = output;
        smooth(work, output, 2);
    };
    const double two_grid_factor =
        estimate_asymptotic_factor(Q.rows(), stationary, apply_two_grid, no_projection,
                                   0x93c1ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_block_two_grid = [&](const num::vec<double> &input, num::vec<double> &output) {
        block_two_grid.apply_system(input, system_image);
        block_two_grid.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double block_two_grid_factor = estimate_asymptotic_factor(
        Q.rows(), stationary, apply_block_two_grid, no_projection,
        0xa4d2ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_vcycle = [&](const num::vec<double> &input, num::vec<double> &output) {
        multigrid.apply_system(input, system_image);
        multigrid.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double vcycle_factor =
        estimate_asymptotic_factor(Q.rows(), stationary, apply_vcycle, no_projection,
                                   0xb4d3ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_block_vcycle = [&](const num::vec<double> &input, num::vec<double> &output) {
        block_multigrid.apply_system(input, system_image);
        block_multigrid.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double block_vcycle_factor =
        estimate_asymptotic_factor(Q.rows(), stationary, apply_block_vcycle, no_projection,
                                   0xc5e6ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_block_wcycle = [&](const num::vec<double> &input, num::vec<double> &output) {
        block_wcycle.apply_system(input, system_image);
        block_wcycle.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double block_wcycle_factor =
        estimate_asymptotic_factor(Q.rows(), stationary, apply_block_wcycle, no_projection,
                                   0xd6f8ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_wcycle = [&](const num::vec<double> &input, num::vec<double> &output) {
        wcycle.apply_system(input, system_image);
        wcycle.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double wcycle_factor =
        estimate_asymptotic_factor(Q.rows(), stationary, apply_wcycle, no_projection,
                                   0xd5e7ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_variable_vcycle = [&](const num::vec<double> &input, num::vec<double> &output) {
        variable_vcycle.apply_system(input, system_image);
        variable_vcycle.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double variable_vcycle_factor = estimate_asymptotic_factor(
        Q.rows(), stationary, apply_variable_vcycle, no_projection,
        0xf719ULL + problem.cells_per_axis * 31 + problem.population);

    result result{problem,
                  multigrid.space().n_voxels(),
                  Q.rows(),
                  multigrid.n_levels(),
                  multigrid.grid_complexity(),
                  multigrid.operator_complexity(),
                  smoothing_one,
                  smoothing_two,
                  block_smoothing,
                  two_grid_factor,
                  block_two_grid_factor,
                  vcycle_factor,
                  block_vcycle_factor,
                  block_wcycle_factor,
                  wcycle_factor,
                  variable_vcycle_factor};
    results.push_back(result);
    std::cout << problem.dimension << ',' << problem.cells_per_axis << ',' << result.voxels << ','
              << problem.population << ',' << result.states << ',' << problem.shift << ','
              << result.levels << ',' << result.grid_complexity << ',' << result.operator_complexity
              << ',' << result.smoothing_factor_one << ',' << result.smoothing_factor_two << ','
              << result.block_smoothing_factor << ',' << result.two_grid_factor << ','
              << result.block_two_grid_factor << ',' << result.vcycle_factor << ','
              << result.block_vcycle_factor << ',' << result.block_wcycle_factor << ','
              << result.wcycle_factor << ',' << result.variable_vcycle_factor << '\n';
    std::cout.flush();
}

void dispatch(const benchmark_case &problem) {
    if (problem.dimension == 1)
        return run_case<1>(problem);
    if (problem.dimension == 2)
        return run_case<2>(problem);
    throw std::invalid_argument("RDME multigrid analysis dimension must be 1 or 2");
}

num::array<benchmark_case> quick_cases() {
    return {{1, 8, 1, 0.5}, {1, 8, 2, 0.5}, {1, 8, 4, 0.5}, {1, 16, 2, 0.5}, {2, 4, 2, 0.5}};
}

num::array<benchmark_case> paper_cases() {
    num::array<benchmark_case> cases;
    for (int population : {1, 2, 3, 4, 6})
        cases.push_back({1, 8, population, 0.5});
    for (int population : {1, 2, 3, 4})
        cases.push_back({1, 16, population, 0.5});
    for (int population : {1, 2, 3})
        cases.push_back({1, 32, population, 0.5});
    for (int population : {1, 2})
        cases.push_back({1, 64, population, 0.5});
    cases.push_back({1, 128, 1, 0.5});

    for (int population : {1, 2, 3, 4, 6})
        cases.push_back({2, 4, population, 0.5});
    for (int population : {1, 2, 3})
        cases.push_back({2, 8, population, 0.5});
    return cases;
}

void save_plot() {
    num::plt::subplot(1, 3);
    for (int population : {1, 2, 3, 4, 6}) {
        num::series values;
        for (const result &result : results) {
            if (result.problem.dimension == 1 && result.problem.population == population)
                values.store(static_cast<double>(result.states), result.smoothing_factor_one);
        }
        std::ranges::sort(values, {}, &num::plot_point::first);
        if (!values.empty())
            num::plt::plot(values, "k = " + std::to_string(population), "linespoints");
    }
    num::plt::title("High-frequency weighted-Jacobi smoothing");
    num::plt::xlabel("configuration-space states");
    num::plt::ylabel("measured smoothing factor");
    num::plt::semilogx();
    num::plt::ylim(0.0, 1.05);
    num::plt::legend("top left");
    num::plt::next();

    for (int population : {1, 2, 3, 4, 6}) {
        num::series values;
        for (const result &result : results) {
            if (result.problem.dimension == 1 && result.problem.population == population)
                values.store(static_cast<double>(result.states), result.vcycle_factor);
        }
        std::ranges::sort(values, {}, &num::plot_point::first);
        if (!values.empty())
            num::plt::plot(values, "k = " + std::to_string(population), "linespoints");
    }
    num::plt::title("Jacobi V-cycle error contraction");
    num::plt::xlabel("configuration-space states");
    num::plt::ylabel("asymptotic convergence factor");
    num::plt::semilogx();
    num::plt::ylim(0.0, 1.05);
    num::plt::legend("top left");
    num::plt::next();

    for (int population : {1, 2, 3, 4, 6}) {
        num::series values;
        for (const result &result : results) {
            if (result.problem.dimension == 1 && result.problem.population == population)
                values.store(static_cast<double>(result.states), result.block_vcycle_factor);
        }
        std::ranges::sort(values, {}, &num::plot_point::first);
        if (!values.empty())
            num::plt::plot(values, "k = " + std::to_string(population), "linespoints");
    }
    num::plt::title("Aggregate-block V-cycle error contraction");
    num::plt::xlabel("configuration-space states");
    num::plt::ylabel("asymptotic convergence factor");
    num::plt::semilogx();
    num::plt::ylim(0.0, 1.05);
    num::plt::legend("top left");
    num::plt::savefig("rdme_multigrid_analysis.png");
}

} // namespace

int main(int argc, char **argv) {
    bool quick = false;
    bool plot = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--quick")
            quick = true;
        else if (value == "--plot")
            plot = true;
        else if (value == "--help") {
            std::cerr << "usage: rdme_multigrid_analysis [--quick] [--plot]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << value << '\n';
            return 2;
        }
    }

    std::cout << std::setprecision(10);
    std::cout << "dimension,cells_per_axis,voxels,population,states,shift,levels,"
                 "grid_complexity,operator_complexity,smoothing_factor_one,"
                 "smoothing_factor_two,block_smoothing_factor,two_grid_factor,"
                 "block_two_grid_factor,vcycle_factor,block_vcycle_factor,block_wcycle_factor,"
                 "wcycle_factor,"
                 "variable_vcycle_factor\n";
    for (const benchmark_case &problem : quick ? quick_cases() : paper_cases())
        dispatch(problem);
    if (plot)
        save_plot();
    return 0;
}
