#include "utils.cuh"
#include <stdexcept>
#include <vector>

namespace pmpp {
template <typename T>
__global__ void vec_add_kernel(const T* A, const T* B, T* C, size_t N) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

template <typename T>
__global__ void vec_mul_kernel(T* u, T x, size_t N) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    u[i] *= x;
  }
}

template<typename T>
__global__ void matmul_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < K) {
    float dot = 0;
    for (int k = 0; k < N; k++) {
      dot += A[i*N + k] * B[k*M + j];
    }
    C[i*K + j] = dot;
  }
}

template<typename T>
__global__ void matadd_kernel(T* A, T* B, T* C, size_t N, size_t M) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < N && j < M) {
    int idx = i*M + j;
    C[idx] = A[idx] + B[idx];
  }
}

template <typename T>
std::vector<T> vec_mul(std::vector<T>& u, T x) {
  size_t N = u.size();

  T* u_d;

  CUDA_CHECK(cudaMalloc((void**)&u_d, N * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(u_d, u.data(), N * sizeof(T), cudaMemcpyHostToDevice));

  vec_mul_kernel<<<1, N>>>(u_d, x, N);

  CUDA_CHECK(cudaMemcpy(u.data(), u_d, N * sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(u_d));

  return u;
}

template <typename T>
std::vector<T> vec_add(const std::vector<T>& u, const std::vector<T>& v) {
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

  constexpr int block_size = 256;
  int num_blocks = (N + block_size - 1) / block_size;

  vec_add_kernel<<<num_blocks, block_size>>>(u_d, v_d, out_d, N);

  CUDA_CHECK(cudaMemcpy(out.data(), out_d, N * sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(u_d));
  CUDA_CHECK(cudaFree(v_d));
  CUDA_CHECK(cudaFree(out_d));

  return out;
}

template <typename T>
std::vector<std::vector<T>> matadd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
  if (A.size() == 0 || B.size() == 0) {
    return {};
  }

  size_t N = A.size(), M = A[0].size();
  if (B.size() != N || B[0].size() != M) {
    throw std::invalid_argument("Matrix dimension mismatch A: " + std::to_string(N) + "x" + std::to_string(M) + 
    ", B: " + std::to_string(B.size()) + "x" + std::to_string(B[0].size()));
  }

  std::vector<std::vector<T>> C(N, std::vector<T>(M));
  T *A_d, *B_d, *C_d;

  size_t bytes = N * M * sizeof(T);
  CUDA_CHECK(cudaMalloc((void**)&A_d, bytes));
  CUDA_CHECK(cudaMalloc((void**)&B_d, bytes));
  CUDA_CHECK(cudaMalloc((void**)&C_d, bytes));

  CUDA_CHECK(cudaMemcpy(A_d, A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int block_size = 256;
  int num_blocks = (N + block_size - 1) / block_size;
  matadd_kernel<<<num_blocks, block_size>>>(A_d, B_d, C_d, N, M);

  CUDA_CHECK(cudaMemcpy(C.data(), C_d, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  return C;
}

template <typename T>
std::vector<std::vector<T>> matmul(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
  if (A.size() == 0 || B.size() == 0) {
    throw std::invalid_argument("Cannot multiply by an empty matrix");
  }

  size_t M = A.size(), N = A[0].size(), K = B[0].size();
  if (B.size() != N) {
    throw std::invalid_argument("Matrix dimension mismatch A: " + std::to_string(M) + "x" + std::to_string(N) + 
    ", B: " + std::to_string(B.size()) + "x" + std::to_string(K));
  }

  std::vector<std::vector<T>> C(M, std::vector<T>(K));
  T *A_d, *B_d, *C_d;

  CUDA_CHECK(cudaMalloc((void**)&A_d, M * N * sizeof(T)));
  CUDA_CHECK(cudaMalloc((void**)&B_d, N * K * sizeof(T)));
  CUDA_CHECK(cudaMalloc((void**)&C_d, M * K * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(A_d, A.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B.data(), N * K * sizeof(T), cudaMemcpyHostToDevice));

  constexpr int block_size = 256;
  int num_blocks = (N + block_size - 1) / block_size;
  matmul_kernel<<<num_blocks, block_size>>>(A_d, B_d, C_d, M, N, K);

  CUDA_CHECK(cudaMemcpy(C.data(), C_d, M * K * sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  return C;
}



} // namespace pmpp
