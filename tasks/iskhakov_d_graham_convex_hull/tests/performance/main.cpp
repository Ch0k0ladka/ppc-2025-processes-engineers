#include <gtest/gtest.h>

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"
#include "iskhakov_d_graham_convex_hull/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_graham_convex_hull/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace iskhakov_d_graham_convex_hull {

class IskhakovDRunGrahamConvexHullPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(IskhakovDRunGrahamConvexHullPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, IskhakovDRunGrahamConvexHullMPI, IskhakovDRunGrahamConvexHullSEQ>(
        PPC_SETTINGS_iskhakov_d_graham_convex_hull);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IskhakovDRunGrahamConvexHullPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IskhakovDRunGrahamConvexHullPerfTests, kGtestValues, kPerfTestName);

}  // namespace  iskhakov_d_graham_convex_hull
