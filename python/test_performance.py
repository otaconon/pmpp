import pmpp_bindings as pmpp
import pytest

TEST_SIZES = [10_000, 100_000, 1_000_000]
TEST_SIZES_SMALL = [10, 100, 1000, 1000]

def gpu_vec_add(u, v):
  return u + v

@pytest.mark.parametrize("N", TEST_SIZES)
def test_gpu_vec_add_benchmark(benchmark, N):
  u = pmpp.FloatArray([i for i in range(N)])
  v = pmpp.VectorArray([i for i in range(N)])
  
  def run_and_wait():
    res = gpu_vec_add(u, v)
    pmpp.synchronize()
    return res
    
  res = benchmark(run_and_wait)
  assert(list(res) == [a + b for a, b in zip(u, v)])
  
@pytest.mark.parametrize("N", TEST_SIZES)
def test_cpu_vec_add_benchmark(benchmark, N):
  u = [i for i in range(N)]
  v = [i for i in range(N)]
  
  def run_and_wait():
    res = pmpp.cpu_vec_add(u, v)
    pmpp.synchronize()
    return res
    
  res = benchmark(run_and_wait)
  assert(list(res) == [a + b for a, b in zip(u, v)])
  
@pytest.mark.parametrize("N", TEST_SIZES_SMALL)
def test_gpu_mat_add_benchmark(benchmark, N):
  A = pmpp.VectorArray([[i for i in range(N)] for _ in range(N)])
  B = pmpp.VectorArray([[i for i in range(N)] for _ in range(N)])
  
  def run_and_wait():
    res = A + B
    pmpp.synchronize()
    return res
    
  res = benchmark(run_and_wait)
  assert(list(res) == [[a + b for a, b in zip(row, col)] for row, col in zip(A, B)])