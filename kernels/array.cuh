#pragma once

#include <algorithm>
#include <execution>
#include <stdexcept>
#include <string>
#include <vector>

#include "array_kernels.cuh"
#include "utils.cuh"

namespace pmpp {

template <typename T>
struct is_std_vector : std::false_type {};

template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template <typename T>
struct array {
  array(const std::vector<T>& values)
      : values{values} {
  }

  operator std::vector<T>() const {
    return values;
  }

  T& operator[](size_t idx) {
    return values[idx];
  }

  const T& operator[](size_t idx) const {
    return values[idx];
  }

  friend bool operator==(const array<T>& lhs, const array<T>& rhs) {
    return lhs.values == rhs.values;
  }

  friend bool operator<(const array<T>& lhs, const array<T>& rhs) {
    return lhs.values < rhs.values;
  }

  bool operator<(const array<T>& rhs) {
    return values < rhs.values;
  }

  array<T>& operator+=(const array<T>& rhs) {
    values = vec_add(values, rhs.values);
    return *this;
  }

  array<T>& operator*=(const T& rhs) {
    values = vec_mul(values, rhs);
    return *this;
  }

  friend array<T> operator+(array<T> lhs, const array<T>& rhs) {
    lhs += rhs;
    return lhs;
  }

  friend array<T> operator*(array<T> lhs, const T& rhs) {
    lhs *= rhs;
    return lhs;
  }

  friend array<T> operator*(const T& lhs, const array<T>& rhs) {
    return rhs * lhs;
  }

  std::vector<T> values;
};
template <typename T>
array<T> cpu_vec_add(std::vector<T>& u, std::vector<T>& v) {
  size_t N = u.size();
  if (v.size() != N) {
    throw std::invalid_argument("Input vectors must have the same size " + std::to_string(N));
  }

  array<T> out{std::vector<T>(N)};
  std::transform(std::execution::par, u.begin(), u.end(), v.begin(), out.values.begin(), std::plus<T>());

  return out;
}

template <typename U>
struct array<std::vector<U>> {
  using T = std::vector<U>;

  std::vector<T> values;

  T& operator[](size_t idx) {
    return values[idx];
  }

  const T& operator[](size_t idx) const {
    return values[idx];
  }

  friend bool operator==(const array<T>& lhs, const array<T>& rhs) {
    return lhs.values == rhs.values;
  }

  friend bool operator<(const array<T>& lhs, const array<T>& rhs) {
    return lhs.values < rhs.values;
  }

  bool operator<(const array<T>& rhs) {
    return values < rhs.values;
  }

  array<T>& operator+=(const array<T>& rhs) {
    values = matadd(values, rhs.values);
    return *this;
  }

  array<T>& operator*=(const array<T>& rhs) {
    values = matmul(values, rhs.values);
    return *this;
  }

  friend array<T> operator+(array<T> lhs, const array<T>& rhs) {
    lhs += rhs;
    return lhs;
  }

  friend array<T> operator*(array<T> lhs, const array<T>& rhs) {
    lhs *= rhs;
    return lhs;
  }
};
}; // namespace pmpp