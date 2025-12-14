include(GoogleTest)
add_executable(pmpp_test test_array.cu)
target_link_libraries(pmpp_test PRIVATE gtest_main pmpp_kernels)
gtest_discover_tests(pmpp_test)

