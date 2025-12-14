#pragma once

#include <algorithm>
#include <execution>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

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
  size_t rows, cols;
  T* data;

public:
  Tensor(size_t rows, size_t cols, Allocation alloc = Allocation::Host)
      : alloc{alloc},
        byteSize{rows * cols * sizeof(T)},
        rows{rows},
        cols{cols},
        data{nullptr} {
    CUDA_CHECK(cudaMallocHost((void**)&data, byteSize));
  }

  Tensor(std::initializer_list<T> list, Allocation alloc = Allocation::Host)
    : alloc{alloc} {
    rows = 1;
    cols = list.size();
    byteSize = rows * sizeof(T);
    CUDA_CHECK(cudaMallocHost((void**)&data, byteSize));
      
    size_t idx = 0;
    for (auto& val : list) {
      data[idx] = val;
      idx++;
    }
  }

  ~Tensor() {
    if (data) {
      cudaFreeHost(data);
    }
  }

  Tensor(const Tensor& other) 
    : rows(other.rows), cols(other.cols), alloc(other.alloc), byteSize(other.byteSize) {
    CUDA_CHECK(cudaMallocHost((void**)&data, byteSize));
    CUDA_CHECK(cudaMemcpy(data, other.data, byteSize, cudaMemcpyHostToHost));
  }

  Tensor& operator=(const Tensor& other) {
    if (this == &other) {
      return *this;
    }
    
    if (data) {
      cudaFreeHost(data);
    }
    
    rows = other.rows;
    cols = other.cols;
    byteSize = other.byteSize;
    alloc = other.alloc;
    
    CUDA_CHECK(cudaMallocHost((void**)&data, byteSize));
    CUDA_CHECK(cudaMemcpy(data, other.data, byteSize, cudaMemcpyHostToHost));
    
    return *this;
  }

  T* Data() {
    return data;
  }

  T& Get(size_t row, size_t col) {
    return data[row*cols + col];
  }

  T& operator[](size_t idx) {
    return data[idx];
  }

  const T& operator[](size_t idx) const {
    return data[idx];
  }

  friend bool operator==(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
      return false;
    }

    for (size_t i = 0; i < lhs.rows; i++) {
      for (size_t j = 0; j < lhs.cols; j++) {
        if (lhs.data[i*lhs.cols + j] != rhs.data[i*lhs.cols + j]) {
          return false;
        }
      }
    }

    return true;
  }

  friend bool operator!=(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator<(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return lhs.data < rhs.data;
  }

  bool operator<(const Tensor<T>& rhs) {
    return data < rhs.data;
  }

  Tensor<T>& operator+=(const Tensor<T>& rhs) {
    T* newData = vec_add(data, rhs.data, rows * cols, rhs.rows * rhs.cols);
    cudaFreeHost(data);
    data = newData; 
    return *this;
  }

  Tensor<T>& operator*=(const T& rhs) {
    //data = vec_mul(data, rhs);
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
    os << "[";
    for (size_t i = 0; i < tensor.rows; i++) {
      os << "[";
      for (size_t j = 0; j < tensor.cols; j++) {
        os << tensor.data[i*tensor.cols + j] << ", ";
      }
      os << "]";
    }
    os << "]";
    return os;
  }

private:
  static T* vec_add(const T* u, const T* v, size_t N, size_t M) {
    if (N != M) {
      throw std::invalid_argument("Invalid shapes for tensor addition: " + std::to_string(N) + ", " + std::to_string(M));
    }

    T* out;
    CUDA_CHECK(cudaMallocHost((void**)&out, N*M*sizeof(T)));
    T *u_d, *v_d, *out_d;

    CUDA_CHECK(cudaMalloc((void**)&u_d, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&v_d, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&out_d, N * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(u_d, u, N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_d, v, N * sizeof(T), cudaMemcpyHostToDevice));

    constexpr int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    vec_add_kernel<<<num_blocks, block_size>>>(u_d, v_d, out_d, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out, out_d, N * sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(u_d));
    CUDA_CHECK(cudaFree(v_d));
    CUDA_CHECK(cudaFree(out_d));

    return out;
  }
};
}; // namespace pmpp