#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"
#include "iskhakov_d_graham_convex_hull/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_graham_convex_hull/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_graham_convex_hull {

class IskhakovDGrahamConvexHullFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    (void)test_param;
    static int test_counter = 0;
    ++test_counter;
    return "Test_" + std::to_string(test_counter);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const auto &[input, expected_result] = params;

    input_data_ = input;
    expected_result_ = expected_result;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    for (const auto &point : output_data) {
      bool found = false;
      for (const auto &input_point : input_data_) {
        if (std::abs(point.x - input_point.x) < 1e-9 && std::abs(point.y - input_point.y) < 1e-9) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_;
};

namespace {

TestType CreateTestData(const std::vector<Point> &input, const std::vector<Point> &expected) {
  return std::make_tuple(input, expected);
}

}  // namespace

const std::array<TestType, 16> kTestParam = {
    CreateTestData(std::vector<Point>{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
                   std::vector<Point>{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),

    CreateTestData(std::vector<Point>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}},
                   std::vector<Point>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}),

    CreateTestData(std::vector<Point>{{0.0, 0.0}, {2.0, 0.0}, {2.0, 2.0}, {0.0, 2.0}, {1.0, 1.0}},
                   std::vector<Point>{{0.0, 0.0}, {2.0, 0.0}, {2.0, 2.0}, {0.0, 2.0}}),

    CreateTestData(
        std::vector<Point>{
            {0.0, 0.0}, {3.0, 0.0}, {3.0, 3.0}, {0.0, 3.0}, {1.0, 1.0}, {2.0, 1.0}, {1.0, 2.0}, {2.0, 2.0}},
        std::vector<Point>{{0.0, 0.0}, {3.0, 0.0}, {3.0, 3.0}, {0.0, 3.0}}),

    CreateTestData(std::vector<Point>{{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}},
                   std::vector<Point>{{0.0, 0.0}, {3.0, 0.0}}),

    CreateTestData(std::vector<Point>{{0.0, 0.0}, {0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}},
                   std::vector<Point>{{0.0, 0.0}, {0.0, 3.0}}),

    CreateTestData(
        std::vector<Point>{
            {0.0, 0.0}, {4.0, 0.0}, {2.0, 2.0}, {1.0, 1.0}, {3.0, 1.0}, {0.0, 4.0}, {4.0, 4.0}, {2.0, 5.0}},
        std::vector<Point>{{0.0, 0.0}, {4.0, 0.0}, {4.0, 4.0}, {2.0, 5.0}, {0.0, 4.0}}),

    CreateTestData(std::vector<Point>{{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}}, std::vector<Point>{{1.0, 1.0}}),

    CreateTestData(
        std::vector<Point>{{0.0, 0.0}, {0.5, 0.0}, {1.0, 0.0}, {1.5, 0.0}, {2.0, 0.0}, {2.0, 0.5}, {2.0, 1.0},
                           {2.0, 1.5}, {2.0, 2.0}, {1.5, 2.0}, {1.0, 2.0}, {0.5, 2.0}, {0.0, 2.0}, {0.0, 1.5},
                           {0.0, 1.0}, {0.0, 0.5}, {0.5, 0.5}, {1.5, 0.5}, {0.5, 1.5}, {1.5, 1.5}},
        std::vector<Point>{{0.0, 0.0}, {2.0, 0.0}, {2.0, 2.0}, {0.0, 2.0}}),

    CreateTestData(std::vector<Point>{{3.0, 0.0},  {2.85, 0.93},   {2.43, 1.77},   {1.77, 2.43},   {0.93, 2.85},
                                      {0.0, 3.0},  {-0.93, 2.85},  {-1.77, 2.43},  {-2.43, 1.77},  {-2.85, 0.93},
                                      {-3.0, 0.0}, {-2.85, -0.93}, {-2.43, -1.77}, {-1.77, -2.43}, {-0.93, -2.85},
                                      {0.0, -3.0}, {0.93, -2.85},  {1.77, -2.43},  {2.43, -1.77},  {2.85, -0.93},
                                      {0.0, 0.0},  {1.0, 1.0},     {-1.0, 1.0},    {1.0, -1.0},    {-1.0, -1.0}},
                   std::vector<Point>{{3.0, 0.0},  {2.85, 0.93},   {2.43, 1.77},   {1.77, 2.43},   {0.93, 2.85},
                                      {0.0, 3.0},  {-0.93, 2.85},  {-1.77, 2.43},  {-2.43, 1.77},  {-2.85, 0.93},
                                      {-3.0, 0.0}, {-2.85, -0.93}, {-2.43, -1.77}, {-1.77, -2.43}, {-0.93, -2.85},
                                      {0.0, -3.0}, {0.93, -2.85},  {1.77, -2.43},  {2.43, -1.77},  {2.85, -0.93}}),

    CreateTestData(std::vector<Point>{{0.0, 4.0},  {1.2, 1.2},  {4.0, 0.0},  {1.2, -1.2},  {0.0, -4.0},  {-1.2, -1.2},
                                      {-4.0, 0.0}, {-1.2, 1.2}, {2.5, 2.5},  {2.5, -2.5},  {-2.5, -2.5}, {-2.5, 2.5},
                                      {0.0, 2.0},  {1.0, 1.0},  {2.0, 0.0},  {1.0, -1.0},  {0.0, -2.0},  {-1.0, -1.0},
                                      {-2.0, 0.0}, {-1.0, 1.0}, {0.5, 0.5},  {1.5, 0.5},   {0.5, 1.5},   {-0.5, 0.5},
                                      {-1.5, 0.5}, {0.5, -0.5}, {1.5, -0.5}, {-0.5, -0.5}, {-1.5, -0.5}, {0.0, 0.0}},
                   std::vector<Point>{{0.0, 4.0},
                                      {1.2, 1.2},
                                      {2.5, 2.5},
                                      {4.0, 0.0},
                                      {2.5, -2.5},
                                      {1.2, -1.2},
                                      {0.0, -4.0},
                                      {-1.2, -1.2},
                                      {-2.5, -2.5},
                                      {-4.0, 0.0},
                                      {-2.5, 2.5},
                                      {-1.2, 1.2}}),

    CreateTestData(std::vector<Point>{{5.0, 5.0}, {10.0, 10.0}, {0.0, 10.0}, {5.0, 15.0}},
                   std::vector<Point>{{5.0, 5.0}, {10.0, 10.0}, {5.0, 15.0}, {0.0, 10.0}}),

    CreateTestData(std::vector<Point>{{0.0, 0.0}, {3.0, 0.0}, {3.0, 3.0}, {0.0, 3.0}, {1.0, 1.0}, {2.0, 2.0}},
                   std::vector<Point>{{0.0, 0.0}, {3.0, 0.0}, {3.0, 3.0}, {0.0, 3.0}}),

    CreateTestData(
        std::vector<Point>{{0.0, 0.0}, {4.0, 0.0}, {4.0, 4.0}, {0.0, 4.0}, {2.0, 2.0}, {1.0, 3.0}, {3.0, 1.0}},
        std::vector<Point>{{0.0, 0.0}, {4.0, 0.0}, {4.0, 4.0}, {0.0, 4.0}}),

    CreateTestData(std::vector<Point>{{0.0, 0.0}, {2.0, 0.0}, {4.0, 0.0}, {2.0, 2.0}, {4.0, 4.0}, {0.0, 4.0}},
                   std::vector<Point>{{0.0, 0.0}, {4.0, 0.0}, {4.0, 4.0}, {0.0, 4.0}}),

    CreateTestData(std::vector<Point>{{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}},
                   std::vector<Point>{{1.0, 1.0}, {5.0, 5.0}})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IskhakovDGrahamConvexHullMPI, InType>(
                                               kTestParam, PPC_SETTINGS_iskhakov_d_graham_convex_hull),
                                           ppc::util::AddFuncTask<IskhakovDGrahamConvexHullSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_iskhakov_d_graham_convex_hull));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IskhakovDGrahamConvexHullFuncTests::PrintFuncTestName<IskhakovDGrahamConvexHullFuncTests>;

INSTANTIATE_TEST_SUITE_P(GrahamConvexHullFuncTests, IskhakovDGrahamConvexHullFuncTests, kGtestValues, kPerfTestName);

TEST_P(IskhakovDGrahamConvexHullFuncTests, RunFuncTests) {
  ExecuteTest(GetParam());
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IskhakovDGrahamConvexHullFuncTests);

}  // namespace iskhakov_d_graham_convex_hull
