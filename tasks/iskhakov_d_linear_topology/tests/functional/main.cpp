#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"
#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_linear_topology {

class IskhakovDLinearTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &input = std::get<0>(test_param);
    const auto &output = std::get<1>(test_param);
    int process_count = std::get<1>(output);

    return "head_" + std::to_string(input.head_process) + "_tail_" + std::to_string(input.tail_process) + "_data_" +
           std::to_string(input.data.size()) + "_processes_" + std::to_string(process_count);
  }

 protected:
  TestType test_params_;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    test_params_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(test_params_);
    expected_output_ = std::get<1>(test_params_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &actual_result = std::get<0>(output_data);
    int actual_processes = std::get<1>(output_data);

    bool is_under_mpirun = ppc::util::IsUnderMpirun();

    if (!is_under_mpirun) {
      const auto &expected_result = std::get<0>(expected_output_);
      int expected_processes = std::get<1>(expected_output_);

      if (actual_processes != expected_processes) {
        return false;
      }

      if (!actual_result.delivered) {
        return false;
      }

      if (actual_result.data != expected_result.data) {
        return false;
      }

      if (actual_result.head_process != input_data_.head_process) {
        return false;
      }

      if (actual_result.tail_process != input_data_.tail_process) {
        return false;
      }

    } else {
      int proc_rank{};
      int proc_nums{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);

      if (actual_processes != proc_nums) {
        return false;
      }

      if (actual_result.head_process != input_data_.head_process) {
        return false;
      }

      if (actual_result.tail_process != input_data_.tail_process) {
        return false;
      }

      if (input_data_.head_process >= proc_nums) {
        return false;
      }

      if (input_data_.tail_process >= proc_nums) {
        return false;
      }

      bool is_target_process = (proc_rank == input_data_.head_process) || (proc_rank == input_data_.tail_process);

      bool should_have_data = (input_data_.head_process == input_data_.tail_process)
                                  ? (proc_rank == input_data_.head_process)
                                  : is_target_process;

      return should_have_data ? (actual_result.delivered && actual_result.data == input_data_.data)
                              : (!actual_result.delivered && actual_result.data.empty());
    }

    return true;
  }

  InType GetTestInputData() final {
    bool is_under_mpirun = ppc::util::IsUnderMpirun();

    if (is_under_mpirun) {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

      if (proc_rank == input_data_.head_process) {
        return input_data_;
      } else {
        Message empty_input;
        empty_input.head_process = input_data_.head_process;
        empty_input.tail_process = input_data_.tail_process;
        empty_input.data = std::vector<int>{};
        empty_input.delivered = false;
        return empty_input;
      }
    } else {
      return input_data_;
    }
  }
};

class IskhakovDLinearTopologyMpiTests : public IskhakovDLinearTopologyFuncTests {
 protected:
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "MPI tests are not under mpirun\n";
      GTEST_SKIP();
    }

    int proc_rank{};
    int proc_nums{};
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);

    IskhakovDLinearTopologyFuncTests::SetUp();

    bool adapted = false;
    if (input_data_.head_process >= proc_nums) {
      input_data_.head_process = proc_nums - 1;
      adapted = true;
    }
    if (input_data_.tail_process >= proc_nums) {
      input_data_.tail_process = proc_nums - 1;
      adapted = true;
    }

    std::get<0>(expected_output_).head_process = input_data_.head_process;
    std::get<0>(expected_output_).tail_process = input_data_.tail_process;
    std::get<1>(expected_output_) = proc_nums;

    if (adapted && proc_rank == 0) {
      std::cout << "Adapted test: head_process=" << input_data_.head_process
                << ", tail_process=" << input_data_.tail_process << " for " << proc_nums << " processes\n";
    }

    int test_params[3] = {input_data_.head_process, input_data_.tail_process,
                          static_cast<int>(input_data_.data.size())};
    MPI_Bcast(test_params, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (proc_rank != 0) {
      input_data_.head_process = test_params[0];
      input_data_.tail_process = test_params[1];

      if (input_data_.data.empty()) {
        input_data_.data.resize(test_params[2]);
        for (int vector_filling_step = 0; vector_filling_step < test_params[2]; ++vector_filling_step) {
          input_data_.data[vector_filling_step] = vector_filling_step + 1;
        }
      }
      input_data_.delivered = false;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
};

class IskhakovDLinearTopologySeqTests : public IskhakovDLinearTopologyFuncTests {
 protected:
  void SetUp() override {
    if (ppc::util::IsUnderMpirun()) {
      std::cerr << "SEQ tests should not run under mpirun\n";
      GTEST_SKIP();
    }
    IskhakovDLinearTopologyFuncTests::SetUp();
  }
};

namespace {

TEST_P(IskhakovDLinearTopologySeqTests, SeqTests) {
  ExecuteTest(GetParam());
}

TEST_P(IskhakovDLinearTopologyMpiTests, MpiTests) {
  ExecuteTest(GetParam());
}

Message CreateMessage(int head, int tail, int data_size, bool delivered) {
  Message msg;
  msg.head_process = head;
  msg.tail_process = tail;
  msg.delivered = delivered;
  msg.data.clear();
  if (data_size > 0) {
    msg.data.resize(data_size);
    for (int vector_filling_step = 0; vector_filling_step < data_size; ++vector_filling_step) {
      msg.data[vector_filling_step] = vector_filling_step + 1;
    }
  }
  return msg;
}

const std::array<TestType, 2> kSeqParam = {
    TestType{CreateMessage(0, 0, 5, false), OutType{CreateMessage(0, 0, 5, true), 1}},
    TestType{CreateMessage(0, 0, 10, false), OutType{CreateMessage(0, 0, 10, true), 1}}};

const std::array<TestType, 14> kMpiParam = {
    TestType{CreateMessage(0, 0, 5, false), OutType{CreateMessage(0, 0, 5, true), 1}},
    TestType{CreateMessage(0, 0, 10, false), OutType{CreateMessage(0, 0, 10, true), 1}},

    TestType{CreateMessage(0, 1, 15, false), OutType{CreateMessage(0, 1, 10, true), 2}},
    TestType{CreateMessage(1, 0, 20, false), OutType{CreateMessage(1, 0, 20, true), 2}},

    TestType{CreateMessage(0, 2, 25, false), OutType{CreateMessage(0, 2, 15, true), 3}},
    TestType{CreateMessage(2, 0, 30, false), OutType{CreateMessage(2, 0, 20, true), 3}},
    TestType{CreateMessage(1, 2, 35, false), OutType{CreateMessage(1, 2, 15, true), 3}},
    TestType{CreateMessage(2, 1, 40, false), OutType{CreateMessage(2, 1, 20, true), 3}},

    TestType{CreateMessage(0, 3, 45, false), OutType{CreateMessage(0, 3, 15, true), 4}},
    TestType{CreateMessage(3, 0, 50, false), OutType{CreateMessage(3, 0, 20, true), 4}},
    TestType{CreateMessage(1, 3, 55, false), OutType{CreateMessage(1, 3, 15, true), 4}},
    TestType{CreateMessage(3, 1, 60, false), OutType{CreateMessage(3, 1, 20, true), 4}},
    TestType{CreateMessage(2, 3, 65, false), OutType{CreateMessage(2, 3, 15, true), 4}},
    TestType{CreateMessage(3, 2, 70, false), OutType{CreateMessage(3, 2, 20, true), 4}}};

const auto kSeqTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IskhakovDLinearTopologySEQ, InType>(kSeqParam, PPC_SETTINGS_iskhakov_d_linear_topology));

const auto kMpiTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IskhakovDLinearTopologyMPI, InType>(kMpiParam, PPC_SETTINGS_iskhakov_d_linear_topology));

const auto kSeqGtestValues = ppc::util::ExpandToValues(kSeqTasksList);
const auto kMpiGtestValues = ppc::util::ExpandToValues(kMpiTasksList);

const auto kFuncTestName = IskhakovDLinearTopologyFuncTests::PrintFuncTestName<IskhakovDLinearTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(SeqTests, IskhakovDLinearTopologySeqTests, kSeqGtestValues, kFuncTestName);
INSTANTIATE_TEST_SUITE_P(MpiTests, IskhakovDLinearTopologyMpiTests, kMpiGtestValues, kFuncTestName);

}  // namespace

}  // namespace iskhakov_d_linear_topology
