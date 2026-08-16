/// @file multi_index.hpp
/// @brief MultiIndex: fixed-length integer tuple used as a state label in FSP.
///
/// A MultiIndex represents a discrete state of a chemical system
/// (e.g. molecule counts per species). It supports element-wise
/// arithmetic needed for reaction stoichiometry and is hashable for
/// use in hash maps.
///
/// Storage: inline int array (no heap allocation).
/// kMaxDim = 8 covers all practical CME systems.
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>

namespace fsp {

/// @brief Fixed-dimension integer tuple for representing discrete states.
///
/// Stored as a plain int[kMaxDim] array with a 1-byte size field.
/// No heap allocation; safe to copy/swap cheaply inside hash table slots.
struct MultiIndex {
  static constexpr std::size_t kMaxDim = 8;

  int8_t n_ = 0;
  int8_t _pad[3] = {}; // align data_ to 4 bytes
  int data_[kMaxDim] = {};

  MultiIndex() = default;

  MultiIndex(std::initializer_list<int> il)
      : n_(static_cast<int8_t>(il.size())) {
    assert(il.size() <= kMaxDim);
    std::size_t i = 0;
    for (int v : il)
      data_[i++] = v;
  }

  explicit MultiIndex(std::size_t n, int val = 0)
      : n_(static_cast<int8_t>(n)) {
    assert(n <= kMaxDim);
    for (std::size_t i = 0; i < n; ++i)
      data_[i] = val;
  }

  std::size_t size() const {
    return static_cast<std::size_t>(n_);
  }

  int& operator[](std::size_t i) {
    return data_[i];
  }
  int operator[](std::size_t i) const {
    return data_[i];
  }

  // Range iteration (supports range-for and std algorithms)
  int* begin() {
    return data_;
  }
  int* end() {
    return data_ + n_;
  }
  const int* begin() const {
    return data_;
  }
  const int* end() const {
    return data_ + n_;
  }

  bool operator==(const MultiIndex& o) const {
    if (n_ != o.n_)
      return false;
    for (int8_t i = 0; i < n_; ++i)
      if (data_[i] != o.data_[i])
        return false;
    return true;
  }
  bool operator!=(const MultiIndex& o) const {
    return !(*this == o);
  }

  /// @brief Element-wise addition (stoichiometry application)
  MultiIndex operator+(const MultiIndex& o) const {
    MultiIndex r(size());
    for (int8_t i = 0; i < n_; ++i)
      r.data_[i] = data_[i] + o.data_[i];
    return r;
  }
};

} // namespace fsp

// --- std::hash specialization for MultiIndex --------------------------------
namespace std {
template <> struct hash<fsp::MultiIndex> {
  std::size_t operator()(const fsp::MultiIndex& x) const noexcept {
    std::size_t seed = x.size();
    for (int8_t i = 0; i < x.n_; ++i) {
      seed ^= static_cast<std::size_t>(x.data_[i]) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
} // namespace std
