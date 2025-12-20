#include <gtest/gtest.h>

#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"
#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace iskhakov_d_linear_topology {

class IskhakovDLinearTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const size_t data_size = 100000;

    std::vector<int> test_data(data_size);

    for (size_t i = 0; i < data_size; ++i) {
      test_data[i] = static_cast<int>(i % 1000);
    }

    input_data_ = InType{0, 3, test_data, false};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!output_data.delivered) {
      return false;
    }

    if (output_data.data.size() != input_data_.data.size()) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(IskhakovDLinearTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, IskhakovDLinearTopologyMPI, IskhakovDLinearTopologySEQ>(
    PPC_SETTINGS_iskhakov_d_linear_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IskhakovDLinearTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IskhakovDLinearTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace iskhakov_d_linear_topology
