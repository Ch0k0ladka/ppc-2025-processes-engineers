#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"
#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_linear_topology {

class IskhakovDLinearTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  IskhakovDLinearTopologyPerfTests() = default;

  void SetUp() override {
    auto task_info = std::get<1>(GetParam());
    is_mpi = task_info.find("mpi") != std::string::npos;

    if (is_mpi) {
      int world_size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      int data_size = 25000000;

      input_data.head_process = 0;
      input_data.tail_process = std::min(3, world_size - 1);

      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == input_data.head_process) {
        std::vector<int> data(data_size);
        for (int index_for = 0; index_for < data_size; ++index_for) {
          auto index = static_cast<int64_t>(index_for);
          data[index_for] = static_cast<int>(((index * 13LL + 7LL) % 1000000LL) + 1LL);
        }
        input_data.SetData(std::move(data));
      } else {
        input_data.SetData({});
      }
    } else {
      int data_size = 25000000;
      input_data.head_process = 0;
      input_data.tail_process = 0;

      std::vector<int> data(data_size);
      for (int i = 0; i < data_size; ++i) {
        auto index = static_cast<int64_t>(i);
        data[i] = static_cast<int>(((index * 13LL + 7LL) % 1000000LL) + 1LL);
      }
      input_data.SetData(std::move(data));
    }

    input_data.delivered = false;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &result = output_data;

    if (is_mpi) {
      int world_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

      if (result.head_process != input_data.head_process) {
        return false;
      }

      if (result.tail_process != input_data.tail_process) {
        return false;
      }

      return true;
    }

    if (result.head_process != input_data.head_process) {
      return false;
    }

    if (result.tail_process != input_data.tail_process) {
      return false;
    }

    if (!result.delivered) {
      return false;
    }

    if (result.data.size() != input_data.data.size()) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    if (is_mpi) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == input_data.head_process) {
        return input_data;
      }

      Message empty_input;
      empty_input.head_process = input_data.head_process;
      empty_input.tail_process = input_data.tail_process;
      empty_input.SetData({});
      empty_input.delivered = false;
      return empty_input;
    }

    return input_data;
  }

  Message input_data{};
  bool is_mpi{false};
};

TEST_P(IskhakovDLinearTopologyPerfTests, RunPerfModes) {
  if (is_mpi) {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "MPI perf tests are not under mpirun\n";
      GTEST_SKIP();
    }

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (input_data.tail_process >= world_size) {
      if (world_size > 0) {
        input_data.tail_process = world_size - 1;
      } else {
        input_data.tail_process = 0;
      }
    }

    if (input_data.head_process >= world_size || input_data.tail_process >= world_size) {
      std::cerr << "Head or tail process out of bounds. World size: " << world_size
                << ", head: " << input_data.head_process << ", tail: " << input_data.tail_process << '\n';
      GTEST_SKIP();
    }
  }

  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, IskhakovDLinearTopologyMPI, IskhakovDLinearTopologySEQ>(
    PPC_SETTINGS_iskhakov_d_linear_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IskhakovDLinearTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IskhakovDLinearTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace iskhakov_d_linear_topology
