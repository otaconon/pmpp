#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vec.cuh>

namespace py = pybind11;

PYBIND11_MODULE(pmpp_bindings, m) {
    m.doc() = "Pmpp bindings";
    
    m.def("vec_add", &pmpp::vec_add<float>, "add two vectors");
}