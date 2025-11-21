#include <gtest/gtest.h>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"
#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_trapezoidal_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace iskhakov_d_trapezoidal_integration {

class IskhakovDTrapezoidalIntegrationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  static double TestFunctionPerf(double x) {
    return x * x * x * std::sin(x) + 2.0 * std::cos(x);
  }

  void SetUp() override {
    input_data_.lower_level = 0.0;
    input_data_.top_level = 1.0;
    input_data_.number_steps = 1000000;
    input_data_.function = TestFunctionPerf;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double expected = 1.8600;
    double error = std::abs(output_data - expected) / std::abs(expected);
    return error < 0.01;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(IskhakovDTrapezoidalIntegrationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, IskhakovDTrapezoidalIntegrationMPI, IskhakovDTrapezoidalIntegrationSEQ>(
        PPC_SETTINGS_iskhakov_d_trapezoidal_integration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IskhakovDTrapezoidalIntegrationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IskhakovDTrapezoidalIntegrationPerfTests, kGtestValues, kPerfTestName);

}  // namespace iskhakov_d_trapezoidal_integration
