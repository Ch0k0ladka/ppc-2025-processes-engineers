#include <gtest/gtest.h>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"
#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_trapezoidal_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace iskhakov_d_trapezoidal_integration {

class IskhakovDTrapezoidalIntegrationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
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
