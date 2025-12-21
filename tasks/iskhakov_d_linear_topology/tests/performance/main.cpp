#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"
#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_linear_topology {

class IskhakovDLinearTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    auto task_info = std::get<1>(GetParam());
    is_mpi_ = task_info.find("mpi") != std::string::npos;

    if (is_mpi_) {
      int world_size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      int data_size = 1000000;

      input_data_.head_process = 0;
      input_data_.tail_process = std::min(3, world_size - 1);

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == input_data_.head_process) {
        std::vector<int> data(data_size);
        for (int i = 0; i < data_size; ++i) {
          data[i] = (static_cast<long long>(i) * 13 + 7) % 1000000 + 1;
        }
        input_data_.set_data(std::move(data));
      } else {
        input_data_.set_data({});
      }
    } else {
      int data_size = 1000000;
      input_data_.head_process = 0;
      input_data_.tail_process = 0;

      std::vector<int> data(data_size);
      for (int i = 0; i < data_size; ++i) {
        data[i] = (static_cast<long long>(i) * 13 + 7) % 1000000 + 1;
      }
      input_data_.set_data(std::move(data));
    }

    input_data_.delivered = false;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &result = output_data;

    if (is_mpi_) {
      int world_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

      if (result.head_process != input_data_.head_process) {
        return false;
      }

      if (result.tail_process != input_data_.tail_process) {
        return false;
      }

      return true;
    } else {
      if (result.head_process != input_data_.head_process) {
        return false;
      }

      if (result.tail_process != input_data_.tail_process) {
        return false;
      }

      if (!result.delivered) {
        return false;
      }

      if (result.data.size() != input_data_.data.size()) {
        return false;
      }

      return true;
    }
  }

  InType GetTestInputData() final {
    if (is_mpi_) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == input_data_.head_process) {
        return input_data_;
      } else {
        Message empty_input;
        empty_input.head_process = input_data_.head_process;
        empty_input.tail_process = input_data_.tail_process;
        empty_input.set_data({});
        empty_input.delivered = false;
        return empty_input;
      }
    } else {
      return input_data_;
    }
  }

  Message input_data_;
  bool is_mpi_;
};

TEST_P(IskhakovDLinearTopologyPerfTests, RunPerfModes) {
  if (is_mpi_) {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "MPI perf tests are not under mpirun\n";
      GTEST_SKIP();
    }

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (input_data_.tail_process >= world_size) {
      if (world_size > 0) {
        input_data_.tail_process = world_size - 1;
      } else {
        input_data_.tail_process = 0;
      }
    }

    if (input_data_.head_process >= world_size || input_data_.tail_process >= world_size) {
      std::cerr << "Head or tail process out of bounds. World size: " << world_size
                << ", head: " << input_data_.head_process << ", tail: " << input_data_.tail_process << "\n";
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
