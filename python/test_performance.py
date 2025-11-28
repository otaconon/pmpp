import pmpp_bindings as pmpp
import pytest

TEST_SIZES = [10_000, 100_000, 1_000_000]

def gpu_vec_add(u, v):
  return u + v

@pytest.mark.parametrize("N", TEST_SIZES)
def test_gpu_vec_add_benchmark(benchmark, N):
  u = pmpp.array([i for i in range(N)])
  v = pmpp.array([i for i in range(N)])
  
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