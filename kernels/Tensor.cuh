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
class Tensor {
private:
  Allocation alloc;
  size_t byteSize;
  T* values;

public:
  Tensor(size_t rows, size_t cols, Allocation alloc = Allocation::Host)
      : alloc{alloc},
        byteSize{rows * cols * sizeof(T)},
        values{nullptr} {
    values = cudaMallocHost(values, byteSize);
  }

  Tensor& operator=(const std::vector<T>& other) {
    
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
}; // namespace pmpp