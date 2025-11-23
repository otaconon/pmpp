#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>

#include "utils.cuh"

namespace pmpp {
template <typename T>
__global__ void vec_add_kernel(const T *A, const T *B, T *C, size_t N) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

template<typename T>
std::vector<T> vec_add(std::vector<T>& u, std::vector<T>& v) {
  size_t N = u.size();

  if (v.size() != N) {
    throw std::invalid_argument("Input vectors must have size " + std::to_string(N));
  }

  std::vector<T> out(N);
  T *u_d, *v_d, *out_d;

  CUDA_CHECK(cudaMalloc((void**)&u_d, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc((void**)&v_d, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc((void**)&out_d, N * sizeof(T)));

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