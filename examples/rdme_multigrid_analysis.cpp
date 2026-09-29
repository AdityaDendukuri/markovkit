#include "core/types.hpp"
#include "plot/plot.hpp"
#include "rdme/rdme.hpp"
#include "structures/graph/structured_grid.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

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

void normalize(num::vec &x, num::view<const double> stationary) {
    const double norm = weighted_norm(x, stationary);
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::runtime_error("RDME multigrid analysis encountered a zero error vector");
    for (double &value : x)
        value /= norm;
}

num::vec stationary_distribution(const rdme::diffusion_state_space &space) {
    const auto &layout = space.layout();
    num::vec stationary(space.size(), 0.0);
    for (num::idx state = 0; state < space.size(); ++state) {
        double log_probability = 0.0;
        for (std::size_t species = 0; species < layout.species_count(); ++species) {
            const int total = space.species_totals()[species];
            log_probability +=
                std::lgamma(static_cast<double>(total) + 1.0) -
                static_cast<double>(total) * std::log(static_cast<double>(layout.voxel_count()));
            for (std::size_t voxel = 0; voxel < layout.voxel_count(); ++voxel) {
                log_probability -= std::lgamma(
                    static_cast<double>(space[state][layout.index(voxel, species)]) + 1.0);
            }
        }
        stationary[state] = std::exp(log_probability);
    }
    double mass = 0.0;
    for (double value : stationary)
        mass += value;
    for (double &value : stationary)
        value /= mass;
    return stationary;
}

template <typename Transfer>
void remove_coarse_component(const Transfer &transfer, num::vec &fine, num::vec &coarse,
                             num::vec &lifted) {
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
        num::vec error(size, 0.0);
        num::vec next(size, 0.0);
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
    num::structured_grid_graph<D> graph(dimensions, spacing);
    rdme::voxel_domain domain(std::move(graph), num::array<double>{1.0});
    const num::array<int> totals{problem.population};

    rdme::population_multigrid<D> multigrid(domain, totals, problem.shift);
    rdme::population_multigrid<D> block_multigrid(domain, totals, problem.shift, 2, 2, 2.0 / 3.0, 4,
                                                  32, 1, 0,
                                                  rdme::population_smoother::aggregate_block);
    rdme::population_multigrid<D> block_wcycle(domain, totals, problem.shift, 2, 2, 2.0 / 3.0, 4,
                                               32, 2, 0,
                                               rdme::population_smoother::aggregate_block);
    rdme::population_multigrid<D> wcycle(domain, totals, problem.shift, 2, 2, 2.0 / 3.0, 4, 32, 2);
    rdme::population_multigrid<D> variable_vcycle(domain, totals, problem.shift, 2, 2, 2.0 / 3.0, 4,
                                                  32, 1, 4);
    rdme::population_multigrid<D> two_grid(domain, totals, problem.shift, 2, 2, 2.0 / 3.0, 1, 2);
    rdme::population_multigrid<D> block_two_grid(domain, totals, problem.shift, 2, 2, 2.0 / 3.0, 1,
                                                 2, 1, 0,
                                                 rdme::population_smoother::aggregate_block);
    if (!two_grid.has_coarse_level())
        throw std::logic_error("RDME two-grid analysis case has no coarse level");

    const auto &generator = multigrid.fine_generator();
    const auto &transfer = two_grid.transfer();
    const num::vec stationary = stationary_distribution(generator.state_space());
    num::vec system_image(generator.rows(), 0.0);
    num::vec work(generator.rows(), 0.0);
    num::vec coarse(transfer.coarse_size(), 0.0);
    num::vec lifted(generator.rows(), 0.0);
    num::vec coarse_residual(transfer.coarse_size(), 0.0);
    num::vec coarse_correction(transfer.coarse_size(), 0.0);
    num::vec zero(generator.rows(), 0.0);
    constexpr double omega = 2.0 / 3.0;

    const auto smooth = [&](const num::vec &input, num::vec &output, int sweeps) {
        output = input;
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            multigrid.apply_system(output, system_image);
            for (num::idx state = 0; state < output.size(); ++state) {
                const double diagonal = problem.shift + generator.exit_rates()[state];
                work[state] = output[state] - omega * system_image[state] / diagonal;
            }
            std::swap(output, work);
        }
    };
    const auto high_pass = [&](num::vec &error) {
        remove_coarse_component(transfer, error, coarse, lifted);
    };
    const auto no_projection = [](num::vec &) {};

    const double smoothing_one = estimate_asymptotic_factor(
        generator.rows(), stationary,
        [&](const num::vec &input, num::vec &output) { smooth(input, output, 1); }, high_pass,
        0x51a7ULL + problem.cells_per_axis * 31 + problem.population);
    const double smoothing_two = estimate_asymptotic_factor(
        generator.rows(), stationary,
        [&](const num::vec &input, num::vec &output) { smooth(input, output, 2); }, high_pass,
        0x72b9ULL + problem.cells_per_axis * 31 + problem.population);
    const double block_smoothing = estimate_asymptotic_factor(
        generator.rows(), stationary,
        [&](const num::vec &input, num::vec &output) {
            output = input;
            block_multigrid.relax(zero, output, 1);
        },
        high_pass, 0x83baULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_two_grid = [&](const num::vec &input, num::vec &output) {
        smooth(input, output, 2);
        two_grid.apply_system(output, system_image);
        transfer.restrict(system_image, coarse_residual);
        two_grid.coarse_level()->apply_block(coarse_residual, coarse_correction);
        transfer.prolong(coarse_correction, lifted);
        for (num::idx state = 0; state < output.size(); ++state)
            output[state] -= lifted[state];
        work = output;
        smooth(work, output, 2);
    };
    const double two_grid_factor =
        estimate_asymptotic_factor(generator.rows(), stationary, apply_two_grid, no_projection,
                                   0x93c1ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_block_two_grid = [&](const num::vec &input, num::vec &output) {
        block_two_grid.apply_system(input, system_image);
        block_two_grid.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double block_two_grid_factor = estimate_asymptotic_factor(
        generator.rows(), stationary, apply_block_two_grid, no_projection,
        0xa4d2ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_vcycle = [&](const num::vec &input, num::vec &output) {
        multigrid.apply_system(input, system_image);
        multigrid.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double vcycle_factor =
        estimate_asymptotic_factor(generator.rows(), stationary, apply_vcycle, no_projection,
                                   0xb4d3ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_block_vcycle = [&](const num::vec &input, num::vec &output) {
        block_multigrid.apply_system(input, system_image);
        block_multigrid.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double block_vcycle_factor =
        estimate_asymptotic_factor(generator.rows(), stationary, apply_block_vcycle, no_projection,
                                   0xc5e6ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_block_wcycle = [&](const num::vec &input, num::vec &output) {
        block_wcycle.apply_system(input, system_image);
        block_wcycle.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double block_wcycle_factor =
        estimate_asymptotic_factor(generator.rows(), stationary, apply_block_wcycle, no_projection,
                                   0xd6f8ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_wcycle = [&](const num::vec &input, num::vec &output) {
        wcycle.apply_system(input, system_image);
        wcycle.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double wcycle_factor =
        estimate_asymptotic_factor(generator.rows(), stationary, apply_wcycle, no_projection,
                                   0xd5e7ULL + problem.cells_per_axis * 31 + problem.population);

    const auto apply_variable_vcycle = [&](const num::vec &input, num::vec &output) {
        variable_vcycle.apply_system(input, system_image);
        variable_vcycle.apply(system_image, work);
        for (num::idx state = 0; state < input.size(); ++state)
            output[state] = input[state] - work[state];
    };
    const double variable_vcycle_factor = estimate_asymptotic_factor(
        generator.rows(), stationary, apply_variable_vcycle, no_projection,
        0xf719ULL + problem.cells_per_axis * 31 + problem.population);

    result result{problem,
                  generator.domain().n_voxels(),
                  generator.rows(),
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
