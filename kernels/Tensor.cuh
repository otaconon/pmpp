#pragma once

#include <algorithm>
#include <execution>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor_kernels.cuh"
#include "utils.cuh"

namespace pmpp {
template <typename T>
struct is_std_vector : std::false_type {};

template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

enum class Allocation {
  Host,
  Device
};

template <typename T>
struct Tensor {
  Allocation alloc;
  size_t byteSize;
  std::vector<T> values;

  Tensor(std::initializer_list<T> init, Allocation alloc = Allocation::Host)
      : values{init},
      alloc{alloc},
      byteSize{init.size() * sizeof(T)} {
  }
  Tensor(const std::vector<T>& values, Allocation alloc = Allocation::Host) : values{values} {}

  operator std::vector<T>() const {
    return values;
  }

  T& operator[](size_t idx) {
    return values[idx];
  }

  const T& operator[](size_t idx) const {
    return values[idx];
  }

  friend bool operator==(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return lhs.values == rhs.values;
  }

  friend bool operator!=(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator<(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return lhs.values < rhs.values;
  }

  bool operator<(const Tensor<T>& rhs) {
    return values < rhs.values;
  }

  Tensor<T>& operator+=(const Tensor<T>& rhs) {
    values = vec_add(values, rhs.values);
    return *this;
  }

  Tensor<T>& operator*=(const T& rhs) {
    values = vec_mul(values, rhs);
    return *this;
  }

  friend Tensor<T> operator+(Tensor<T> lhs, const Tensor<T>& rhs) {
    lhs += rhs;
    return lhs;
  }

  friend Tensor<T> operator*(Tensor<T> lhs, const T& rhs) {
    lhs *= rhs;
    return lhs;
  }

  friend Tensor<T> operator*(const T& lhs, const Tensor<T>& rhs) {
    return rhs * lhs;
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
    os << "[ ";
    for (const auto& val : tensor.values) {
      os << val << " ";
    }
    os << "]";
    return os;
  }
};

template <typename T>
Tensor<T> cpu_vec_add(std::vector<T>& u, std::vector<T>& v) {
  size_t N = u.size();
  if (v.size() != N) {
    throw std::invalid_argument("Input vectors must have the same size " + std::to_string(N));
  }

  Tensor<T> out{std::vector<T>(N)};
  std::transform(std::execution::par, u.begin(), u.end(), v.begin(), out.values.begin(), std::plus<T>());

  return out;
}

template <typename U>
struct Tensor<std::vector<U>> {
  using T = std::vector<U>;

  std::vector<T> values;

  T& operator[](size_t idx) {
    return values[idx];
  }

  const T& operator[](size_t idx) const {
    return values[idx];
  }

  friend bool operator==(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return lhs.values == rhs.values;
  }

  friend bool operator<(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return lhs.values < rhs.values;
  }

  bool operator<(const Tensor<T>& rhs) {
    return values < rhs.values;
  }

  Tensor<T>& operator+=(const Tensor<T>& rhs) {
    values = matadd(values, rhs.values);
    return *this;
  }

  Tensor<T>& operator*=(const Tensor<T>& rhs) {
    values = matmul(values, rhs.values);
    return *this;
  }

  friend Tensor<T> operator+(Tensor<T> lhs, const Tensor<T>& rhs) {
    lhs += rhs;
    return lhs;
  }

  friend Tensor<T> operator*(Tensor<T> lhs, const Tensor<T>& rhs) {
    lhs *= rhs;
    return lhs;
  }
};
}; // namespace pmpp