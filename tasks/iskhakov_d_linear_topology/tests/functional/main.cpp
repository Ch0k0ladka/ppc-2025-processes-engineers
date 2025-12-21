#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
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

    const auto &expected_result = std::get<0>(expected_output_);
    int expected_processes = std::get<1>(expected_output_);

    bool is_under_mpirun = ppc::util::IsUnderMpirun();

    if (!is_under_mpirun) {
      if (actual_processes != expected_processes) {
        return false;
      }

      if (!actual_result.delivered || actual_result.data != expected_result.data) {
        return false;
      }

      if (actual_result.head_process != input_data_.head_process ||
          actual_result.tail_process != input_data_.tail_process) {
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

      if (actual_result.head_process != input_data_.head_process ||
          actual_result.tail_process != input_data_.tail_process) {
        return false;
      }

      if (input_data_.head_process >= proc_nums || input_data_.tail_process >= proc_nums) {
        return false;
      }

      if (input_data_.head_process == input_data_.tail_process) {
        if (proc_rank == input_data_.head_process) {
          if (!actual_result.delivered || actual_result.data != input_data_.data) {
            return false;
          }
        } else {
          if (actual_result.delivered || !actual_result.data.empty()) {
            return false;
          }
        }
      } else {
        if (proc_rank == input_data_.head_process || proc_rank == input_data_.tail_process) {
          if (!actual_result.delivered || actual_result.data != input_data_.data) {
            return false;
          }
        } else {
          if (actual_result.delivered || !actual_result.data.empty()) {
            return false;
          }
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

    IskhakovDLinearTopologyFuncTests::SetUp();

    int proc_rank{};
    int proc_nums{};
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);

    if (input_data_.head_process >= proc_nums || input_data_.tail_process >= proc_nums) {
      if (proc_rank == 0) {
        std::cerr << "Test requires at least " << std::max(input_data_.head_process, input_data_.tail_process) + 1
                  << " processes, but only " << proc_nums << " available.\n";
      }
      GTEST_SKIP();
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
  msg.data.resize(data_size);
  for (int i = 0; i < data_size; ++i) {
    msg.data[i] = i + 1;
  }
  return msg;
}

const std::array<TestType, 2> kSeqParam = {
    TestType{CreateMessage(0, 0, 5, false), OutType{CreateMessage(0, 0, 5, true), 1}},
    TestType{CreateMessage(0, 0, 10, false), OutType{CreateMessage(0, 0, 10, true), 1}}};

const std::array<TestType, 6> kMpiParam = {
    TestType{CreateMessage(0, 0, 5, false), OutType{CreateMessage(0, 0, 5, true), 1}},
    TestType{CreateMessage(0, 0, 10, false), OutType{CreateMessage(0, 0, 10, true), 1}},
    TestType{CreateMessage(0, 1, 10, false), OutType{CreateMessage(0, 1, 10, true), 2}},
    TestType{CreateMessage(0, 1, 20, false), OutType{CreateMessage(0, 1, 20, true), 2}},
    TestType{CreateMessage(0, 2, 15, false), OutType{CreateMessage(0, 2, 15, true), 3}},
    TestType{CreateMessage(0, 3, 20, false), OutType{CreateMessage(0, 3, 20, true), 4}}};

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
