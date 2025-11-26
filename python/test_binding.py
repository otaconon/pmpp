import pmpp_bindings as pmpp
import pytest

def test_vec_add():
  vec = pmpp.array([1, 2, 3, 4])
  res = vec + vec
  assert res == pmpp.array([2.0, 4.0, 6.0, 8.0])
