add_library(pmpp_kernels STATIC Tensor.cu)

target_include_directories(pmpp_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pmpp_kernels PUBLIC CUDA::cudart_static)

target_compile_features(pmpp_kernels PUBLIC cxx_std_17)
set_target_properties(pmpp_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)