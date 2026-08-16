/// @file else/options.hpp
/// @brief Options, criteria, and shedding stats for FESS/ELSE solvers.
#pragma once

#include <string>
#include <vector>

namespace markovkit::else_solver {

enum class ShedCriterion { Schur, CutTime, CutNorm };

struct FESSOptions {
  int capacity = 60;
  int expansion_depth = 1;
  ShedCriterion criterion = ShedCriterion::Schur;
  double cut_threshold = 1.0e-3;
  int cut_norm_samples = 40;
  double eta_threshold = 1.0e-3;
  int shed_batch_size = 1;
  double atol = 1.0e-12;
};

struct FESSSheddingStats {
  int removed = 0;
  int threshold_cuts = 0;
  int capacity_cuts = 0;
  double estimated_relative_loss = 0.0;
  double actual_relative_loss = 0.0;
  double maximum_single_cut_loss = 0.0;
  int solves = 0;
};

} // namespace markovkit::else_solver
