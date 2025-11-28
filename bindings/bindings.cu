#include <array.cuh>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <utils.cuh>

namespace py = pybind11;

PYBIND11_MODULE(pmpp_bindings, m) {
  m.doc() = "Pmpp bindings";
  m.def("synchronize", &pmpp::sync_device, "Wait for gpu to finish");

  py::class_<pmpp::array<float>>(m, "array")
      .def(py::init<const std::vector<float>&>())
      .def(py::self == py::self)
      .def(py::self < py::self)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self * float())
      .def(float() * py::self)
      .def(py::self *= float())
      .def("to_list", [](const pmpp::array<float>& a) -> std::vector<float> {
        return a.values;
      })
      .def("__len__", [](const pmpp::array<float>& a) {
        return a.values.size();
      })
      .def("__getitem__", [](const pmpp::array<float>& a, size_t i) {
        if (i >= a.values.size())
          throw py::index_error();
        return a[i];
      });

  m.def("cpu_vec_add", &pmpp::cpu_vec_add<float>, py::arg("u"), py::arg("v"));
}