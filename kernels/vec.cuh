#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.cuh"

namespace pmpp {

template <typename T>
std::vector<T> vec_mul(std::vector<T> &u, T x);

template <typename T>
std::vector<T> vec_add(std::vector<T> &u, std::vector<T> &v);

template <typename T>
class array {
public:
  array(const std::vector<T> &values)
      : values{values} {
  }

  array<T> &operator+=(const array<T> &rhs) {
    values = vec_add(values, rhs.values);
    return *this;
  }

  array<T> &operator*=(const T &rhs) {
    values = vec_mul(values, rhs);
    return *this;
  }

  friend array<T> operator+(array<T> lhs, const array<T> &rhs) {
    lhs += rhs;
    return lhs;
  }

  friend array<T> operator*(array<T> lhs, const T &rhs) {
    lhs *= rhs;
    return lhs;
  }

  friend array<T> operator*(const T &lhs, const array<T> &rhs) {
    return rhs * lhs;
  }

private:
  std::vector<T> values;
};

template <typename T>
__global__ void vec_add_kernel(const T *A, const T *B, T *C, size_t N) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

template <typename T>
__global__ void vec_mul_kernel(T *u, T x, size_t N) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    u[i] *= x;
  }
}

template <typename T>
std::vector<T> vec_mul(std::vector<T> &u, T x) {
  size_t N = u.size();

  T *u_d;

  CUDA_CHECK(cudaMalloc((void **)&u_d, N * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(u_d, u.data(), N * sizeof(T), cudaMemcpyHostToDevice));

  vec_mul_kernel<<<1, N>>>(u_d, x, N);

  CUDA_CHECK(cudaMemcpy(u.data(), u_d, N * sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(u_d));

  return u;
}

template <typename T>
std::vector<T> vec_add(const std::vector<T> &u, const std::vector<T> &v) {
  size_t N = u.size();

  if (v.size() != N) {
    throw std::invalid_argument("Input vectors must have size " + std::to_string(N));
  }

  std::vector<T> out(N);
  T *u_d, *v_d, *out_d;

  CUDA_CHECK(cudaMalloc((void **)&u_d, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc((void **)&v_d, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc((void **)&out_d, N * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(u_d, u.data(), N * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v_d, v.data(), N * sizeof(T), cudaMemcpyHostToDevice));

  vec_add_kernel<<<1, N>>>(u_d, v_d, out_d, N);

  CUDA_CHECK(cudaMemcpy(out.data(), out_d, N * sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(u_d));
  CUDA_CHECK(cudaFree(v_d));
  CUDA_CHECK(cudaFree(out_d));

  return out;
}

}; // namespace pmpp