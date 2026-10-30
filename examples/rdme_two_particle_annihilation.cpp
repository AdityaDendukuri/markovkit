// diffusing A + B -> nothing: Laplace transform of the annihilation time via resolvent solves
#include "core/types.hpp"
#include "linear/expv/expv.hpp"
#include "linear/solvers/gmres.hpp"
#include "plot/plot.hpp"
#include "markovkit.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace markovkit;

namespace {

using clock = std::chrono::steady_clock;

struct benchmark_case {
    int dimension;
    num::idx cells_per_axis;
    double diffusivity;
    double reaction_rate;
    double shift;

    bool operator==(const benchmark_case &) const = default;
};

struct observation {
    num::solver_result solver;
    double setup_ms{};
    double solve_ms{};
    double laplace_transform{};
    double vacuum_transform{};
    double survival_transform{};
    double identity_error{};
    double vacuum_flux_error{};
    std::size_t levels{};
};

struct plot_record {
    benchmark_case problem;
    num::idx states;
    std::string initial_condition;
    std::string method;
    observation observation;
};

num::array<plot_record> plot_records;

double milliseconds(clock::time_point begin, clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

// A + B -> nothing at rate k a b / V, with both species diffusing on a grid
template <std::size_t D>
reaction_diffusion<grid<D>> annihilation(grid<D> cells, double diffusivity, double k) {
    return {voxel_domain(std::move(cells), {diffusivity, diffusivity}),
            {{{-1, -1}, [=](view<const int> n, real volume) { return k * n[0] * n[1] / volume; }}}};
}

template <typename Preconditioner, typename System>
observation solve(const System &system, const Preconditioner &preconditioner, const num::vec<double> &rhs,
                  num::idx vacuum, num::idx joint_begin, num::idx joint_end, double shift,
                  const num::vec<double> &reaction_rates, const num::gmres_options &options,
                  double setup_ms, std::size_t levels) {
    num::vec<double> solution(system.rows(), 0.0);
    const auto begin = clock::now();
    const auto result = num::gmres(system, preconditioner, rhs, solution, options);
    const auto end = clock::now();

    double survival_transform = 0.0;
    double laplace_transform = 0.0;
    for (num::idx state = joint_begin; state < joint_end; ++state) {
        survival_transform += solution[state];
        laplace_transform += reaction_rates[state] * solution[state];
    }
    const double vacuum_transform = shift * solution[vacuum];
    return {result,
            setup_ms,
            milliseconds(begin, end),
            laplace_transform,
            vacuum_transform,
            survival_transform,
            std::abs(laplace_transform + shift * survival_transform - 1.0),
            std::abs(laplace_transform - vacuum_transform),
            levels};
}

template <typename System>
observation solve_plain(const System &system, const num::vec<double> &rhs, num::idx vacuum,
                        num::idx joint_begin, num::idx joint_end, double shift,
                        const num::vec<double> &reaction_rates, const num::gmres_options &options) {
    num::vec<double> solution(system.rows(), 0.0);
    const auto begin = clock::now();
    const auto result = num::gmres(system, rhs, solution, options);
    const auto end = clock::now();

    double survival_transform = 0.0;
    double laplace_transform = 0.0;
    for (num::idx state = joint_begin; state < joint_end; ++state) {
        survival_transform += solution[state];
        laplace_transform += reaction_rates[state] * solution[state];
    }
    const double vacuum_transform = shift * solution[vacuum];
    return {result,
            0.0,
            milliseconds(begin, end),
            laplace_transform,
            vacuum_transform,
            survival_transform,
            std::abs(laplace_transform + shift * survival_transform - 1.0),
            std::abs(laplace_transform - vacuum_transform),
            0};
}

void emit(const benchmark_case &problem, num::idx voxels, num::idx states,
          std::string_view initial_condition, double initial_separation, std::string_view method,
          const observation &observation, double reference_transform, double generator_setup_ms) {
    const double transform_difference =
        observation.solver.converged ? std::abs(observation.laplace_transform - reference_transform)
                                     : std::numeric_limits<double>::quiet_NaN();
    std::cout << problem.dimension << ',' << problem.cells_per_axis << ',' << voxels << ','
              << states << ',' << initial_condition << ',' << initial_separation << ','
              << problem.diffusivity << ',' << problem.reaction_rate << ',' << problem.shift << ','
              << method << ',' << observation.levels << ',' << generator_setup_ms << ','
              << observation.setup_ms << ',' << observation.solve_ms << ','
              << observation.solver.iterations << ',' << observation.solver.residual << ','
              << observation.solver.converged << ',' << observation.laplace_transform << ','
              << observation.vacuum_transform << ',' << observation.survival_transform << ','
              << observation.identity_error << ',' << observation.vacuum_flux_error << ','
              << transform_difference << '\n';
    plot_records.push_back(
        {problem, states, std::string(initial_condition), std::string(method), observation});
}

template <typename XValue, typename YValue, typename Select>
void add_plot_series(std::string label, XValue x_value, YValue y_value, Select select) {
    num::series values;
    for (const auto &record : plot_records) {
        if (select(record))
            values.store(x_value(record), y_value(record));
    }
    std::ranges::sort(values, {}, &num::plot_point::first);
    if (!values.empty())
        num::plt::plot(values, std::move(label), "linespoints");
}

void save_transform_plot() {
    num::plt::subplot(1, 2);
    for (const std::string_view initial_condition : {"colocated", "opposite_corners"}) {
        for (int dimension : {1, 2, 3}) {
            add_plot_series(
                std::to_string(dimension) + "D",
                [](const plot_record &record) {
                    return static_cast<double>(record.problem.cells_per_axis);
                },
                [](const plot_record &record) { return record.observation.laplace_transform; },
                [=](const plot_record &record) {
                    return record.method == "multigrid" &&
                           record.initial_condition == initial_condition &&
                           record.problem.dimension == dimension &&
                           record.problem.diffusivity == 1.0 &&
                           record.problem.reaction_rate == 1.0 && record.problem.shift == 0.5;
                });
        }
        num::plt::title(initial_condition == "colocated" ? "Initially colocated"
                                                         : "Initially separated");
        num::plt::xlabel("cells per axis");
        num::plt::ylabel("E[exp(-s T)], s = 0.5");
        num::plt::semilogx();
        num::plt::ylim(0.0, 1.0);
        num::plt::legend("bottom left");
        if (initial_condition == "colocated")
            num::plt::next();
    }
    num::plt::savefig("rdme_annihilation_transform.png");
}

void save_solver_plot() {
    num::plt::subplot(1, 2);
    for (const std::string_view method : {"plain", "multigrid"}) {
        for (int dimension : {1, 2, 3}) {
            add_plot_series(
                std::to_string(dimension) + "D",
                [](const plot_record &record) { return static_cast<double>(record.states); },
                [](const plot_record &record) {
                    return static_cast<double>(record.observation.solver.iterations);
                },
                [=](const plot_record &record) {
                    return record.method == method &&
                           record.initial_condition == "opposite_corners" &&
                           record.problem.dimension == dimension &&
                           record.problem.diffusivity == 1.0 &&
                           record.problem.reaction_rate == 1.0 && record.problem.shift == 0.5;
                });
        }
        num::plt::title(method == "plain" ? "Plain GMRES" : "Multigrid-preconditioned GMRES");
        num::plt::xlabel("configuration-space states");
        num::plt::ylabel("GMRES iterations");
        num::plt::loglog();
        num::plt::legend("top left");
        if (method == "plain")
            num::plt::next();
    }
    num::plt::savefig("rdme_annihilation_multigrid.png");
}

void save_configuration_density_plot() {
    constexpr idx cells = 32;
    constexpr double spacing = 1.0 / static_cast<double>(cells);
    const auto model = annihilation(grid<1>({cells}, {spacing}), 1.0, 10.0);
    const population_box box(cells, {1, 1});
    const generator Q(box, model);

    state initial_state(2 * cells, 0);
    initial_state[model.index(0, 0)] = 1;
    initial_state[model.index(cells - 1, 1)] = 1;
    vec<real> probability(Q.rows(), 0.0);
    probability[*box.find(initial_state)] = 1.0;
    const idx joint = *box.find_sector({1, 1});

    const num::static_array<double, 4> times{0.002, 0.02, 0.1, 0.5};
    num::static_array<vec<real>, 4> densities;
    num::static_array<double, 4> survival{};
    num::static_array<double, 4> color_max{};
    double current_time = 0.0;
    for (std::size_t snapshot = 0; snapshot < times.size(); ++snapshot) {
        while (current_time < times[snapshot]) {
            const double step = std::min(0.005, times[snapshot] - current_time);
            probability = num::expv(step, Q, probability, 40, 1e-12);
            current_time += step;
        }
        double total_probability = 0.0;
        double minimum_probability = 0.0;
        for (const double mass : probability) {
            total_probability += mass;
            minimum_probability = std::min(minimum_probability, mass);
        }
        if (std::abs(total_probability - 1.0) > 1e-8 || minimum_probability < -1e-10)
            throw std::runtime_error("transient RDME probability check failed");
        densities[snapshot] = vec<real>(cells * cells, 0.0);
        for (idx i = box.sector_begin(joint); i < box.sector_end(joint); ++i) {
            // the one A molecule and the one B molecule
            idx a_voxel = 0, b_voxel = 0;
            for (idx voxel = 0; voxel < cells; ++voxel) {
                a_voxel += voxel * box[i][model.index(voxel, 0)];
                b_voxel += voxel * box[i][model.index(voxel, 1)];
            }
            const double mass = std::max(0.0, probability[i]);
            densities[snapshot][a_voxel * cells + b_voxel] = mass;
            survival[snapshot] += mass;
            color_max[snapshot] = std::max(color_max[snapshot], mass);
        }
        if (snapshot > 0 && survival[snapshot] > survival[snapshot - 1] + 1e-10)
            throw std::runtime_error("annihilation survival probability increased");
    }

    const auto render = [&](int rows, int columns, const std::string &filename) {
        num::plt::subplot(rows, columns);
        for (std::size_t snapshot = 0; snapshot < times.size(); ++snapshot) {
            num::plt::heatmap(densities[snapshot], static_cast<int>(cells), spacing, 0.0,
                              color_max[snapshot]);
            num::plt::colormap(
                "defined (0 '#440154', 0.33 '#31688e', 0.66 '#35b779', 1 '#fde725')");
            std::ostringstream title;
            title << std::fixed << std::setprecision(3) << "t = " << times[snapshot]
                  << "   survival = " << survival[snapshot];
            num::plt::title(title.str());
            num::plt::xlabel("particle A position");
            num::plt::ylabel("particle B position");
            if (snapshot + 1 < times.size())
                num::plt::next();
        }
        num::plt::savefig(filename);
    };
    render(2, 2, "rdme_annihilation_density.png");
    render(1, 4, "rdme_annihilation_density_wide.png");
}

void save_plots() {
    save_transform_plot();
    save_solver_plot();
    save_configuration_density_plot();
    std::cerr << "saved rdme_annihilation_transform.png, rdme_annihilation_multigrid.png, "
                 "rdme_annihilation_density.png, and rdme_annihilation_density_wide.png\n";
}

template <std::size_t D>
void run_case(const benchmark_case &problem) {
    num::static_array<num::idx, D> dimensions{};
    dimensions.fill(problem.cells_per_axis);
    num::static_array<double, D> spacing{};
    spacing.fill(1.0 / static_cast<double>(problem.cells_per_axis));
    const auto model =
        annihilation(grid<D>(dimensions, spacing), problem.diffusivity, problem.reaction_rate);
    const idx voxels = model.n_voxels();

    const auto generator_begin = clock::now();
    const population_box box(voxels, {1, 1});
    const generator Q(box, model);
    const double generator_setup_ms = milliseconds(generator_begin, clock::now());
    const shifted_generator system(Q, problem.shift);

    const idx joint = *box.find_sector({1, 1});
    const idx joint_begin = box.sector_begin(joint);
    const idx joint_end = box.sector_end(joint);
    const idx vacuum = *box.find(state(2 * voxels, 0));
    // the annihilation rate out of each two-particle state
    vec<real> reaction_rates(Q.rows(), 0.0);
    for (idx i = joint_begin; i < joint_end; ++i) {
        for (const auto &jump : Q.jumps(i)) {
            if (jump.to == vacuum) {
                reaction_rates[i] += jump.rate;
            }
        }
    }

    const auto multigrid_begin = clock::now();
    const sector_multigrid<D> multigrid(model.domain, box, problem.shift);
    const double multigrid_setup_ms = milliseconds(multigrid_begin, clock::now());
    const num::gmres_options options{.tolerance = 1e-9,
                                     .max_iterations = 1000,
                                     .restart = std::min<idx>(50, Q.rows())};

    for (const bool separated : {false, true}) {
        state initial(2 * voxels, 0);
        initial[model.index(0, 0)] = 1;
        initial[model.index(separated ? voxels - 1 : 0, 1)] = 1;
        vec<real> rhs(Q.rows(), 0.0);
        rhs[*box.find(initial)] = 1.0;

        const observation plain = solve_plain(system, rhs, vacuum, joint_begin, joint_end,
                                              problem.shift, reaction_rates, options);
        const observation accelerated =
            solve(system, multigrid, rhs, vacuum, joint_begin, joint_end, problem.shift,
                  reaction_rates, options, multigrid_setup_ms, multigrid.max_levels());
        const double reference =
            plain.solver.converged ? plain.laplace_transform : accelerated.laplace_transform;
        const double separation =
            separated ? std::sqrt(static_cast<double>(D)) *
                            (1.0 - 1.0 / static_cast<double>(problem.cells_per_axis))
                      : 0.0;
        const std::string_view initial_condition = separated ? "opposite_corners" : "colocated";
        emit(problem, voxels, Q.rows(), initial_condition, separation, "plain", plain, reference,
             generator_setup_ms);
        emit(problem, voxels, Q.rows(), initial_condition, separation, "multigrid", accelerated,
             reference, generator_setup_ms);
    }
}

void dispatch(const benchmark_case &problem) {
    if (problem.dimension == 1)
        return run_case<1>(problem);
    if (problem.dimension == 2)
        return run_case<2>(problem);
    if (problem.dimension == 3)
        return run_case<3>(problem);
    throw std::invalid_argument("annihilation benchmark dimension must be 1, 2, or 3");
}

num::array<benchmark_case> quick_cases() {
    return {{1, 16, 1.0, 1.0, 0.5}, {2, 4, 1.0, 1.0, 0.5}, {3, 2, 1.0, 1.0, 0.5}};
}

num::array<benchmark_case> paper_cases() {
    num::array<benchmark_case> cases;
    const auto add = [&](int dimension, num::idx cells, double diffusion, double reaction,
                         double shift) {
        const benchmark_case candidate{dimension, cells, diffusion, reaction, shift};
        if (std::find(cases.begin(), cases.end(), candidate) == cases.end())
            cases.push_back(candidate);
    };

    for (num::idx cells : {num::idx{8}, num::idx{16}, num::idx{32}, num::idx{64}})
        add(1, cells, 1.0, 1.0, 0.5);
    for (num::idx cells : {num::idx{4}, num::idx{8}, num::idx{16}})
        add(2, cells, 1.0, 1.0, 0.5);
    for (num::idx cells : {num::idx{2}, num::idx{4}})
        add(3, cells, 1.0, 1.0, 0.5);

    for (int dimension : {1, 2, 3}) {
        const num::idx cells = dimension == 1 ? 32 : (dimension == 2 ? 8 : 4);
        for (double reaction : {0.1, 1.0, 10.0})
            add(dimension, cells, 1.0, reaction, 0.5);
        for (double shift : {0.1, 0.5, 2.0})
            add(dimension, cells, 1.0, 1.0, shift);
    }
    return cases;
}

} // namespace

int main(int argc, char **argv) {
    bool quick = false;
    bool make_plots = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--quick") {
            quick = true;
        } else if (value == "--plot") {
            make_plots = true;
        } else if (value == "--help") {
            std::cerr << "usage: rdme_two_particle_annihilation [--quick] [--plot]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << value << '\n';
            return 2;
        }
    }

    std::cout << std::setprecision(10);
    std::cout << "dimension,cells_per_axis,voxels,states,initial_condition,initial_separation,"
                 "diffusivity,reaction_rate,shift,method,levels,generator_setup_ms,"
                 "preconditioner_setup_ms,solve_ms,iterations,residual,converged,"
                 "annihilation_time_laplace_transform,vacuum_transform,survival_transform,"
                 "probability_identity_error,vacuum_flux_error,absolute_transform_difference\n";
    for (const benchmark_case &problem : quick ? quick_cases() : paper_cases())
        dispatch(problem);
    if (make_plots)
        save_plots();
    return 0;
}
