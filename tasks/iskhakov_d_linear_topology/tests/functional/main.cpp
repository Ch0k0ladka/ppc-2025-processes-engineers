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
    return "head_" + std::to_string(input.head_process) + "_tail_" + std::to_string(input.tail_process) + "_data_" +
           std::to_string(input.data.size());
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
    const auto &actual_result = output_data;
    bool is_under_mpirun = ppc::util::IsUnderMpirun();

    if (!is_under_mpirun) {
      const auto &expected_result = expected_output_;

      if (actual_result.head_process != input_data_.head_process) {
        std::cerr << "SEQ: head_process mismatch" << std::endl;
        return false;
      }

      if (actual_result.tail_process != input_data_.tail_process) {
        std::cerr << "SEQ: tail_process mismatch" << std::endl;
        return false;
      }

      if (!actual_result.delivered) {
        std::cerr << "SEQ: delivered should be true" << std::endl;
        return false;
      }

      if (actual_result.data != expected_result.data) {
        std::cerr << "SEQ: data mismatch" << std::endl;
        return false;
      }

      return true;
    } else {
      int proc_rank{};
      int proc_nums{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);

      if (actual_result.head_process != input_data_.head_process) {
        std::cerr << "MPI[" << proc_rank << "]: head_process mismatch" << std::endl;
        return false;
      }

      if (actual_result.tail_process != input_data_.tail_process) {
        std::cerr << "MPI[" << proc_rank << "]: tail_process mismatch" << std::endl;
        return false;
      }

      bool is_head = (proc_rank == input_data_.head_process);
      bool is_tail = (proc_rank == input_data_.tail_process);
      bool same_process = (input_data_.head_process == input_data_.tail_process);

      if (same_process) {
        if (is_head) {
          if (!actual_result.delivered) {
            std::cerr << "MPI[" << proc_rank << "]: same process should have delivered=true" << std::endl;
            return false;
          }
        } else {
          if (actual_result.delivered) {
            std::cerr << "MPI[" << proc_rank << "]: non-participant should have delivered=false" << std::endl;
            return false;
          }
        }
      } else {
        if (is_head) {
          if (actual_result.delivered) {
            std::cerr << "MPI[" << proc_rank << "]: head should have delivered=false" << std::endl;
            return false;
          }
        } else if (is_tail) {
          if (!actual_result.delivered) {
            std::cerr << "MPI[" << proc_rank << "]: tail should have delivered=true" << std::endl;
            return false;
          }
        } else {
          if (actual_result.delivered) {
            std::cerr << "MPI[" << proc_rank << "]: intermediate should have delivered=false" << std::endl;
            return false;
          }
        }
      }
      if (same_process) {
        if (is_head) {
          if (actual_result.data != input_data_.data) {
            std::cerr << "MPI[" << proc_rank << "]: same process data mismatch" << std::endl;
            return false;
          }
        } else {
          if (!actual_result.data.empty()) {
            std::cerr << "MPI[" << proc_rank << "]: non-participant should have empty data" << std::endl;
            return false;
          }
        }
      } else {
        if (is_tail) {
          if (actual_result.data != input_data_.data) {
            std::cerr << "MPI[" << proc_rank << "]: tail data mismatch" << std::endl;
            return false;
          }
        } else {
          if (!actual_result.data.empty()) {
            std::cerr << "MPI[" << proc_rank << "]: non-tail should have empty data" << std::endl;
            return false;
          }
        }
      }

      return true;
    }
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
        empty_input.set_data({});
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
      GTEST_SKIP() << "MPI tests are not under mpirun";
    }

    int proc_rank{};
    int proc_nums{};
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);

    IskhakovDLinearTopologyFuncTests::SetUp();

    expected_output_ = Message{};
    expected_output_.head_process = -1;
    expected_output_.tail_process = -1;
    expected_output_.set_data({});
    expected_output_.delivered = false;

    bool adapted = false;
    if (input_data_.head_process >= proc_nums) {
      input_data_.head_process = proc_nums - 1;
      adapted = true;
    }
    if (input_data_.tail_process >= proc_nums) {
      input_data_.tail_process = proc_nums - 1;
      adapted = true;
    }

    expected_output_.head_process = input_data_.head_process;
    expected_output_.tail_process = input_data_.tail_process;
    expected_output_.set_data(input_data_.data);
    expected_output_.delivered = true;

    if (proc_rank == 0 && adapted) {
      std::cout << "Adapted test: head_process=" << input_data_.head_process
                << ", tail_process=" << input_data_.tail_process << " for " << proc_nums << " processes\n";
    }

    int test_params[3] = {input_data_.head_process, input_data_.tail_process,
                          static_cast<int>(input_data_.data.size())};
    MPI_Bcast(test_params, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (proc_rank != 0) {
      input_data_.head_process = test_params[0];
      input_data_.tail_process = test_params[1];
      input_data_.set_data({});
      input_data_.delivered = false;
    }

    int data_size = test_params[2];
    if (data_size > 0) {
      if (proc_rank == 0) {
        MPI_Bcast(input_data_.data.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
      } else {
        input_data_.data.resize(data_size);
        MPI_Bcast(input_data_.data.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
      }
    }

    int expected_params[4];
    if (proc_rank == 0) {
      expected_params[0] = expected_output_.head_process;
      expected_params[1] = expected_output_.tail_process;
      expected_params[2] = expected_output_.delivered ? 1 : 0;
      expected_params[3] = static_cast<int>(expected_output_.data.size());
    }

    MPI_Bcast(expected_params, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (proc_rank != 0) {
      expected_output_.head_process = expected_params[0];
      expected_output_.tail_process = expected_params[1];
      expected_output_.delivered = (expected_params[2] != 0);
      expected_output_.data.resize(expected_params[3]);
    }

    if (expected_params[3] > 0) {
      MPI_Bcast(expected_output_.data.data(), expected_params[3], MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
};

class IskhakovDLinearTopologySeqTests : public IskhakovDLinearTopologyFuncTests {
 protected:
  void SetUp() override {
    if (ppc::util::IsUnderMpirun()) {
      GTEST_SKIP() << "SEQ tests skipped under mpirun";
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

  if (data_size > 0) {
    std::vector<int> data(data_size);
    for (int i = 0; i < data_size; ++i) {
      data[i] = i + 1;
    }
    msg.set_data(std::move(data));
  } else {
    msg.set_data({});
  }

  return msg;
}

const std::array<TestType, 2> kSeqParam = {TestType{CreateMessage(0, 0, 5, false), CreateMessage(0, 0, 5, true)},
                                           TestType{CreateMessage(0, 0, 10, false), CreateMessage(0, 0, 10, true)}};

const std::array<TestType, 14> kMpiParam = {TestType{CreateMessage(0, 0, 5, false), CreateMessage(0, 0, 5, true)},
                                            TestType{CreateMessage(0, 0, 10, false), CreateMessage(0, 0, 10, true)},

                                            TestType{CreateMessage(0, 1, 15, false), CreateMessage(0, 1, 15, true)},
                                            TestType{CreateMessage(1, 0, 20, false), CreateMessage(1, 0, 20, true)},

                                            TestType{CreateMessage(0, 2, 25, false), CreateMessage(0, 2, 25, true)},
                                            TestType{CreateMessage(2, 0, 30, false), CreateMessage(2, 0, 30, true)},
                                            TestType{CreateMessage(1, 2, 35, false), CreateMessage(1, 2, 35, true)},
                                            TestType{CreateMessage(2, 1, 40, false), CreateMessage(2, 1, 40, true)},

                                            TestType{CreateMessage(0, 3, 45, false), CreateMessage(0, 3, 45, true)},
                                            TestType{CreateMessage(3, 0, 50, false), CreateMessage(3, 0, 50, true)},
                                            TestType{CreateMessage(1, 3, 55, false), CreateMessage(1, 3, 55, true)},
                                            TestType{CreateMessage(3, 1, 60, false), CreateMessage(3, 1, 60, true)},
                                            TestType{CreateMessage(2, 3, 65, false), CreateMessage(2, 3, 65, true)},
                                            TestType{CreateMessage(3, 2, 70, false), CreateMessage(3, 2, 70, true)}};

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
