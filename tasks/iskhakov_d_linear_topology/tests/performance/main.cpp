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
    int data_size = 100000000;

    input_data_.head_process = 0;
    input_data_.data_size = data_size;

    auto task_info = std::get<1>(GetParam());
    is_mpi_ = task_info.find("mpi") != std::string::npos;

    if (is_mpi_) {
      input_data_.tail_process = 3;
    } else {
      input_data_.tail_process = 0;
    }

    input_data_.delivered = false;

    input_data_.data.resize(data_size);
    for (int vector_filling_step = 0; vector_filling_step < data_size; ++vector_filling_step) {
      input_data_.data[vector_filling_step] = (vector_filling_step * 13 + 7) % 1000000 + 1;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &result = std::get<0>(output_data);
    int processes_number = std::get<1>(output_data);

    if (is_mpi_) {
      int world_size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      if (processes_number != world_size) {
        return false;
      }

      if (result.head_process != input_data_.head_process) {
        return false;
      }

      if (result.tail_process != input_data_.tail_process) {
        return false;
      }

    } else {
      if (processes_number != 1) {
        return false;
      }

      if (result.head_process != input_data_.head_process) {
        return false;
      }

      if (result.tail_process != input_data_.tail_process) {
        return false;
      }

      if (!result.delivered) {
        return false;
      }
    }

    return true;
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
        empty_input.data_size = 0;
        empty_input.data = std::vector<int>{};
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
