#include <gtest/gtest.h>
#include <Tensor.cuh>

TEST(TestTensorComparison, Tensor1D) {
  pmpp::Tensor<float> u{1, 2, 3, 4};
  pmpp::Tensor<float> v{1, 2, 3, 4};
  pmpp::Tensor<float> w{2, 2, 3, 4};

  EXPECT_EQ(u, v);
  EXPECT_NE(u, w);
  EXPECT_LT(u, w);
}

TEST(TestTensorAccess, Tensor1D) {
  pmpp::Tensor<float> u({1.f, 2.f, 3.f, 4.f});
  EXPECT_EQ(u[0], 1);
  EXPECT_EQ(u[3], 4);
}

TEST(TestTensorAddition, Tensor1D) {
  pmpp::Tensor<float> u{1, 2, 3, 4};
  pmpp::Tensor<float> v{1, 2, 3, 4};

  EXPECT_EQ(u + v, (pmpp::Tensor<float>{2, 4, 6, 8}));
}

TEST(TestTensorSubtraction, Tensor1D) {
  pmpp::Tensor<float> u{1, 2, 3, 4};
  pmpp::Tensor<float> v{1, 2, 3, 4};

  //EXPECT_EQ(u, (pmpp::Tensor<float>{2, 4, 6, 8}));
}

TEST(TestTensorMultiplication, Tensor1D) {
  pmpp::Tensor<float> u{1, 2, 3, 4};
  float x = 3;

  EXPECT_EQ(u * x, (pmpp::Tensor<float>{3, 6, 9, 12}));
}

TEST(TestTensorMultiplication, Tensor2D) {
  //pmpp::Tensor<std::vector<float>> u{{1, 2, 3}, {3, 3, 3}, {4, 5, 6}};
  //pmpp::Tensor<std::vector<float>> v{{2, 2, 3}, {4, 3, 5}, {4, 5, 6}};

}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}