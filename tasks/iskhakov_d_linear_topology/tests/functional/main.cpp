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
           std::to_string(input.data_size) + "_processes_" + std::to_string(process_count);
  }

 protected:
  TestType test_params_;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    test_params_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(test_params_);
    expected_output_ = std::get<1>(test_params_);

    if (input_data_.data_size > 0 && input_data_.data.empty()) {
      input_data_.data.resize(input_data_.data_size);
      for (int i = 0; i < input_data_.data_size; ++i) {
        input_data_.data[i] = i + 1;
      }
    }

    auto &expected_msg = std::get<0>(expected_output_);
    if (expected_msg.data_size > 0 && expected_msg.data.empty()) {
      expected_msg.data.resize(expected_msg.data_size);
      for (int i = 0; i < expected_msg.data_size; ++i) {
        expected_msg.data[i] = i + 1;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &actual_result = std::get<0>(output_data);
    int actual_processes = std::get<1>(output_data);

    bool is_under_mpirun = ppc::util::IsUnderMpirun();

    if (!is_under_mpirun) {
      const auto &expected_result = std::get<0>(expected_output_);
      int expected_processes = std::get<1>(expected_output_);

      if (actual_processes != expected_processes) {
        std::cerr << "Process count mismatch: actual=" << actual_processes << ", expected=" << expected_processes
                  << std::endl;
        return false;
      }

      if (!actual_result.delivered) {
        std::cerr << "Message not delivered" << std::endl;
        return false;
      }

      if (actual_result.data != expected_result.data) {
        std::cerr << "Data mismatch" << std::endl;
        return false;
      }

      if (actual_result.head_process != input_data_.head_process) {
        std::cerr << "Head process mismatch: actual=" << actual_result.head_process
                  << ", expected=" << input_data_.head_process << std::endl;
        return false;
      }

      if (actual_result.tail_process != input_data_.tail_process) {
        std::cerr << "Tail process mismatch: actual=" << actual_result.tail_process
                  << ", expected=" << input_data_.tail_process << std::endl;
        return false;
      }

    } else {
      int proc_rank{};
      int proc_nums{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);

      if (actual_processes != proc_nums) {
        std::cerr << "Process count mismatch: actual=" << actual_processes << ", expected=" << proc_nums << std::endl;
        return false;
      }

      if (actual_result.head_process != input_data_.head_process) {
        std::cerr << "Head process mismatch: actual=" << actual_result.head_process
                  << ", expected=" << input_data_.head_process << std::endl;
        return false;
      }

      if (actual_result.tail_process != input_data_.tail_process) {
        std::cerr << "Tail process mismatch: actual=" << actual_result.tail_process
                  << ", expected=" << input_data_.tail_process << std::endl;
        return false;
      }

      if (input_data_.head_process >= proc_nums) {
        std::cerr << "Head process out of range: " << input_data_.head_process << " >= " << proc_nums << std::endl;
        return false;
      }

      if (input_data_.tail_process >= proc_nums) {
        std::cerr << "Tail process out of range: " << input_data_.tail_process << " >= " << proc_nums << std::endl;
        return false;
      }

      bool is_target_process = (proc_rank == input_data_.head_process) || (proc_rank == input_data_.tail_process);

      bool should_have_data = (input_data_.head_process == input_data_.tail_process)
                                  ? (proc_rank == input_data_.head_process)
                                  : is_target_process;

      if (should_have_data) {
        if (!actual_result.delivered) {
          std::cerr << "Process " << proc_rank << " should have data but not delivered" << std::endl;
          return false;
        }
        if (actual_result.data != input_data_.data) {
          std::cerr << "Process " << proc_rank << " data mismatch" << std::endl;
          return false;
        }
      } else {
        if (actual_result.delivered) {
          std::cerr << "Process " << proc_rank << " should not have data but delivered" << std::endl;
          return false;
        }
        if (!actual_result.data.empty()) {
          std::cerr << "Process " << proc_rank << " should not have data but data is not empty" << std::endl;
          return false;
        }
      }
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
        empty_input.data_size = 0;
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

    auto &expected_msg = std::get<0>(expected_output_);
    expected_msg.head_process = input_data_.head_process;
    expected_msg.tail_process = input_data_.tail_process;
    expected_msg.data_size = input_data_.data_size;
    expected_msg.delivered = true;

    expected_msg.data.clear();
    if (input_data_.data_size > 0) {
      expected_msg.data.resize(input_data_.data_size);
      for (int i = 0; i < input_data_.data_size; ++i) {
        expected_msg.data[i] = i + 1;
      }
    }

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
      input_data_.data_size = test_params[2];

      if (input_data_.data.empty() && test_params[2] > 0) {
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

Message CreateMessage(int head, int tail, int data_size, bool delivered) {
  Message msg;
  msg.head_process = head;
  msg.tail_process = tail;
  msg.data_size = data_size;
  msg.delivered = delivered;

  msg.data.clear();
  msg.data.shrink_to_fit();

  if (data_size > 0) {
    msg.data.resize(data_size);
    for (int i = 0; i < data_size; ++i) {
      msg.data[i] = i + 1;
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

    TestType{CreateMessage(0, 1, 15, false), OutType{CreateMessage(0, 1, 15, true), 2}},
    TestType{CreateMessage(1, 0, 20, false), OutType{CreateMessage(1, 0, 20, true), 2}},

    TestType{CreateMessage(0, 2, 25, false), OutType{CreateMessage(0, 2, 25, true), 3}},
    TestType{CreateMessage(2, 0, 30, false), OutType{CreateMessage(2, 0, 30, true), 3}},
    TestType{CreateMessage(1, 2, 35, false), OutType{CreateMessage(1, 2, 35, true), 3}},
    TestType{CreateMessage(2, 1, 40, false), OutType{CreateMessage(2, 1, 40, true), 3}},

    TestType{CreateMessage(0, 3, 45, false), OutType{CreateMessage(0, 3, 45, true), 4}},
    TestType{CreateMessage(3, 0, 50, false), OutType{CreateMessage(3, 0, 50, true), 4}},
    TestType{CreateMessage(1, 3, 55, false), OutType{CreateMessage(1, 3, 55, true), 4}},
    TestType{CreateMessage(3, 1, 60, false), OutType{CreateMessage(3, 1, 60, true), 4}},
    TestType{CreateMessage(2, 3, 65, false), OutType{CreateMessage(2, 3, 65, true), 4}},
    TestType{CreateMessage(3, 2, 70, false), OutType{CreateMessage(3, 2, 70, true), 4}}};

const auto kSeqTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IskhakovDLinearTopologySEQ, InType>(kSeqParam, PPC_SETTINGS_iskhakov_d_linear_topology));

const auto kMpiTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IskhakovDLinearTopologyMPI, InType>(kMpiParam, PPC_SETTINGS_iskhakov_d_linear_topology));

const auto kSeqGtestValues = ppc::util::ExpandToValues(kSeqTasksList);
const auto kMpiGtestValues = ppc::util::ExpandToValues(kMpiTasksList);

const auto kFuncTestName = IskhakovDLinearTopologyFuncTests::PrintFuncTestName<IskhakovDLinearTopologyFuncTests>;

TEST_P(IskhakovDLinearTopologySeqTests, SeqTests) {
  ExecuteTest(GetParam());
}

TEST_P(IskhakovDLinearTopologyMpiTests, MpiTests) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(SeqTests, IskhakovDLinearTopologySeqTests, kSeqGtestValues, kFuncTestName);
INSTANTIATE_TEST_SUITE_P(MpiTests, IskhakovDLinearTopologyMpiTests, kMpiGtestValues, kFuncTestName);

}  // namespace

}  // namespace iskhakov_d_linear_topology
