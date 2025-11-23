#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vec.cuh>
#include <pybind11/operators.h>

namespace py = pybind11;

PYBIND11_MODULE(pmpp_bindings, m) {
    m.doc() = "Pmpp bindings";
    
    py::class_<pmpp::array<float>>(m, "array")
      .def(py::init<const std::vector<float>&>())
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self * float())
      .def(float() * py::self)
      .def(py::self *= float());
}