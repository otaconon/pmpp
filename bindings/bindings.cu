#include <Tensor.cuh>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <utils.cuh>

namespace py = pybind11;

template <typename T>
void bind_pmpp_array(py::module_& m, const std::string& python_name) {
  using ClassType = pmpp::Tensor<T>;
  using ValueType = decltype(std::declval<ClassType>().values);

  py::class_<ClassType>(m, python_name.c_str())
      .def(py::init<const ValueType&>())
      .def(py::self == py::self)
      .def(py::self < py::self)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def("to_list", [](const ClassType& a) {
        return a.values;
      })
      .def("__len__", [](const ClassType& a) {
        return a.values.size();
      })
      .def("__getitem__", [](const ClassType& a, size_t i) {
        if (i >= a.values.size())
          throw py::index_error();
        return a[i];
      });
}

PYBIND11_MODULE(pmpp_bindings, m) {
  m.doc() = "Pmpp bindings";
  m.def("synchronize", &pmpp::sync_device, "Wait for gpu to finish");
  m.def("cpu_vec_add", &pmpp::cpu_vec_add<float>, py::arg("u"), py::arg("v"));

  bind_pmpp_array<float>(m, "FloatArray");
  bind_pmpp_array<std::vector<float>>(m, "VectorArray");
}