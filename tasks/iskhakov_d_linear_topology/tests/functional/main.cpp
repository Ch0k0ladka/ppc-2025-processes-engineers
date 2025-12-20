#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"
#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"

namespace iskhakov_d_linear_topology {

class LinearTopologyFuncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
  }

  std::vector<int> make_data(int count, int start = 1) {
    std::vector<int> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = start + i;
    }
    return data;
  }

  void check_head_tail(const Message &result, const std::vector<int> &expected_data, int head, int tail) {
    if (head == tail) {
      if (rank_ == head) {
        EXPECT_TRUE(result.delivered);
        EXPECT_EQ(result.data, expected_data);
      } else {
        EXPECT_FALSE(result.delivered);
        EXPECT_TRUE(result.data.empty());
      }
    } else {
      if (rank_ == head || rank_ == tail) {
        EXPECT_TRUE(result.delivered);
        EXPECT_EQ(result.data, expected_data);
      } else {
        EXPECT_FALSE(result.delivered);
        EXPECT_TRUE(result.data.empty());
      }
    }

    EXPECT_EQ(result.head_process, head);
    EXPECT_EQ(result.tail_process, tail);
  }

  int rank_;
  int size_;
};

TEST_F(LinearTopologyFuncTest, SingleProcess) {
  MPI_Barrier(MPI_COMM_WORLD);

  int head = 0;
  int tail = 0;
  std::vector<int> test_data = make_data(5);

  Message input;
  input.head_process = head;
  input.tail_process = tail;
  input.data = (rank_ == head) ? test_data : std::vector<int>{};
  input.delivered = false;

  IskhakovDLinearTopologyMPI algorithm(input);

  bool valid = algorithm.Validation();
  if (rank_ == head) {
    EXPECT_TRUE(valid);
  }

  if (valid) {
    algorithm.PreProcessing();
    bool run = algorithm.Run();
    EXPECT_TRUE(run);
    algorithm.PostProcessing();
  }

  auto output = algorithm.GetOutput();
  const auto &result = std::get<0>(output);
  int processes_number = std::get<1>(output);

  check_head_tail(result, test_data, head, tail);
  EXPECT_EQ(processes_number, size_);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(LinearTopologyFuncTest, TwoProcesses) {
  MPI_Barrier(MPI_COMM_WORLD);

  if (size_ < 2) {
    MPI_Barrier(MPI_COMM_WORLD);
    GTEST_SKIP() << "Need at least 2 processes";
  }

  int head = 0;
  int tail = 1;
  std::vector<int> test_data = make_data(10);

  Message input;
  input.head_process = head;
  input.tail_process = tail;
  input.data = (rank_ == head) ? test_data : std::vector<int>{};
  input.delivered = false;

  IskhakovDLinearTopologyMPI algorithm(input);

  bool valid = algorithm.Validation();
  if (rank_ == head) {
    EXPECT_TRUE(valid);
  }

  if (valid) {
    algorithm.PreProcessing();
    bool run = algorithm.Run();
    EXPECT_TRUE(run);
    algorithm.PostProcessing();
  }

  auto output = algorithm.GetOutput();
  const auto &result = std::get<0>(output);
  int processes_number = std::get<1>(output);

  check_head_tail(result, test_data, head, tail);
  EXPECT_EQ(processes_number, size_);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(LinearTopologyFuncTest, ThreeOrMoreProcesses) {
  MPI_Barrier(MPI_COMM_WORLD);

  if (size_ < 3) {
    MPI_Barrier(MPI_COMM_WORLD);
    GTEST_SKIP() << "Need at least 3 processes";
  }

  int head = 0;
  int tail = size_ - 1;
  std::vector<int> test_data = make_data(15);

  Message input;
  input.head_process = head;
  input.tail_process = tail;
  input.data = (rank_ == head) ? test_data : std::vector<int>{};
  input.delivered = false;

  IskhakovDLinearTopologyMPI algorithm(input);

  bool valid = algorithm.Validation();
  if (rank_ == head) {
    EXPECT_TRUE(valid);
  }

  if (valid) {
    algorithm.PreProcessing();
    bool run = algorithm.Run();
    EXPECT_TRUE(run);
    algorithm.PostProcessing();
  }

  auto output = algorithm.GetOutput();
  const auto &result = std::get<0>(output);
  int processes_number = std::get<1>(output);

  check_head_tail(result, test_data, head, tail);
  EXPECT_EQ(processes_number, size_);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(LinearTopologyFuncTest, FourOrMoreProcesses) {
  MPI_Barrier(MPI_COMM_WORLD);

  if (size_ < 4) {
    MPI_Barrier(MPI_COMM_WORLD);
    GTEST_SKIP() << "Need at least 4 processes";
  }

  int head = 0;
  int tail = size_ - 1;
  std::vector<int> test_data = make_data(20);

  Message input;
  input.head_process = head;
  input.tail_process = tail;
  input.data = (rank_ == head) ? test_data : std::vector<int>{};
  input.delivered = false;

  IskhakovDLinearTopologyMPI algorithm(input);

  bool valid = algorithm.Validation();
  if (rank_ == head) {
    EXPECT_TRUE(valid);
  }

  if (valid) {
    algorithm.PreProcessing();
    bool run = algorithm.Run();
    EXPECT_TRUE(run);
    algorithm.PostProcessing();
  }

  auto output = algorithm.GetOutput();
  const auto &result = std::get<0>(output);
  int processes_number = std::get<1>(output);

  check_head_tail(result, test_data, head, tail);
  EXPECT_EQ(processes_number, size_);

  MPI_Barrier(MPI_COMM_WORLD);
}

class LinearTopologySEQTest : public ::testing::Test {};

TEST_F(LinearTopologySEQTest, BasicSEQTest) {
  auto make_data = [](int count) {
    std::vector<int> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = i + 1;
    }
    return data;
  };

  int head = 0;
  int tail = 0;
  std::vector<int> test_data = make_data(10);

  Message input;
  input.head_process = head;
  input.tail_process = tail;
  input.data = test_data;
  input.delivered = false;

  IskhakovDLinearTopologySEQ algorithm(input);

  EXPECT_TRUE(algorithm.Validation());
  algorithm.PreProcessing();
  EXPECT_TRUE(algorithm.Run());
  algorithm.PostProcessing();

  auto output = algorithm.GetOutput();
  const auto &result = std::get<0>(output);
  int processes_number = std::get<1>(output);

  EXPECT_TRUE(result.delivered);
  EXPECT_EQ(result.data, test_data);
  EXPECT_EQ(result.head_process, head);
  EXPECT_EQ(result.tail_process, tail);
  EXPECT_EQ(processes_number, 1);
}

}  // namespace iskhakov_d_linear_topology
