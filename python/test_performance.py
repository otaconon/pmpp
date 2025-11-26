import pmpp_bindings as pmpp
import pytest

def gpu_vec_add(u, v):
  return u + v

def test_gpu_vec_add_benchmark(benchmark):
  N = 1000000
  u = pmpp.array([i for i in range(N)])
  v = pmpp.array([i for i in range(N)])
  
  def run_and_wait():
    gpu_vec_add(u, v)
    pmpp.synchronize()
    
  benchmark(run_and_wait)
  
def test_cpu_vec_add_benchmark(benchmark):
  N = 1000000
  u = [i for i in range(N)]
  v = [i for i in range(N)]
  
  def run_and_wait():
    pmpp.cpu_vec_add(u, v)
    pmpp.synchronize()
    
  benchmark(run_and_wait)