/// @file flat_hash_map.hpp
/// @brief Re-exports FlatHashMap for fsp namespace.
#pragma once

#include <functional>
#include <unordered_map>

namespace fsp {

template <typename Key,
          typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using FlatHashMap = std::unordered_map<Key, Value, Hash, KeyEqual>;

} // namespace fsp
