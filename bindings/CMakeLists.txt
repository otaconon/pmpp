set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG)
pybind11_add_module(pmpp_bindings bindings.cu)

target_link_libraries(pmpp_bindings PRIVATE pmpp_kernels CUDA::cudart_static pybind11::headers)
set_target_properties(pmpp_bindings PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS pmpp_bindings LIBRARY DESTINATION .)