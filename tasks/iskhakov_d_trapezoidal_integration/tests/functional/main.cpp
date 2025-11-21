#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"
#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_trapezoidal_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_trapezoidal_integration {

class IskhakovDTrapezoidalIntegrationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[input, expected_result] = test_param;

    double lower_level = std::get<0>(input);
    double top_level = std::get<1>(input);

    int lower_level_int = static_cast<int>(lower_level);
    int top_level_int = static_cast<int>(top_level);

    return "FROM_" + std::to_string(lower_level_int) + "_TO_" + std::to_string(top_level_int);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const auto &[input, expected_result] = params;

    input_data_ = input;
    expected_result_ = expected_result;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double relative_error = std::abs(output_data - expected_result_) / std::abs(expected_result_);
    return relative_error < 0.01;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double expected_result_;
};

namespace {

double InFunction(double x) {
  return x * x * x * std::sin(x) + 2.0 * std::cos(x);
}

InType CreateTestData(double lower_level, double top_level, int steps) {
  return std::make_tuple(lower_level, top_level, InFunction, steps);
}

TEST_P(IskhakovDTrapezoidalIntegrationFuncTests, TrapezoidalIntegration) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(CreateTestData(0.0, 1.0, 10000), 1.8600),
                                            std::make_tuple(CreateTestData(0.0, 2.0, 20000), 5.6100),
                                            std::make_tuple(CreateTestData(1.0, 3.0, 30000), 10.2953)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IskhakovDTrapezoidalIntegrationMPI, InType>(
                                               kTestParam, PPC_SETTINGS_iskhakov_d_trapezoidal_integration),
                                           ppc::util::AddFuncTask<IskhakovDTrapezoidalIntegrationSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_iskhakov_d_trapezoidal_integration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    IskhakovDTrapezoidalIntegrationFuncTests::PrintFuncTestName<IskhakovDTrapezoidalIntegrationFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, IskhakovDTrapezoidalIntegrationFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace iskhakov_d_trapezoidal_integration
