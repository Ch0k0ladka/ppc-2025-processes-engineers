#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"
#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"

namespace iskhakov_d_linear_topology {

class MpiBarrierGuard {
 public:
  MpiBarrierGuard() {}
  ~MpiBarrierGuard() {
    MPI_Barrier(MPI_COMM_WORLD);
  }
};

class LinearTopologyPerfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
  }

  std::vector<int> make_data(int count) {
    std::vector<int> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = i * 10;
    }
    return data;
  }

  double measure_time(int head, int tail, const std::vector<int> &data) {
    Message input;
    input.head_process = head;
    input.tail_process = tail;
    input.data = (rank_ == head) ? data : std::vector<int>{};
    input.delivered = false;

    IskhakovDLinearTopologyMPI algorithm(input);

    if (!algorithm.Validation()) {
      return -1.0;
    }

    algorithm.PreProcessing();

    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    algorithm.Run();
    algorithm.PostProcessing();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    return duration.count() * 1000;
  }

  int rank_;
  int size_;
};

TEST_F(LinearTopologyPerfTest, SingleProcess) {
  MpiBarrierGuard guard;
  std::vector<int> data = make_data(10000);

  double time_ms = measure_time(0, 0, data);

  if (rank_ == 0) {
    std::cout << "Single process: " << time_ms << " ms for 10000 elements" << std::endl;

    EXPECT_LT(time_ms, 10.0);
  }
}

TEST_F(LinearTopologyPerfTest, TwoProcesses) {
  MpiBarrierGuard guard;
  if (size_ < 2) {
    GTEST_SKIP() << "Need at least 2 processes";
  }

  std::vector<int> data = make_data(5000);

  double time_ms = measure_time(0, 1, data);

  if (rank_ == 0) {
    std::cout << "Two processes: " << time_ms << " ms for 5000 elements" << std::endl;
    EXPECT_LT(time_ms, 100.0);
  }
}

TEST_F(LinearTopologyPerfTest, ThreeOrMoreProcesses) {
  MpiBarrierGuard guard;
  if (size_ < 3) {
    GTEST_SKIP() << "Need at least 3 processes";
  }

  std::vector<int> data = make_data(10000);

  double time_ms = measure_time(0, size_ - 1, data);

  if (rank_ == 0) {
    std::cout << size_ << " processes: " << time_ms << " ms for 10000 elements" << std::endl;

    EXPECT_LT(time_ms, 500.0);
  }
}

TEST_F(LinearTopologyPerfTest, FourOrMoreProcesses) {
  MpiBarrierGuard guard;
  if (size_ < 4) {
    GTEST_SKIP() << "Need at least 4 processes";
  }

  std::vector<int> data = make_data(8000);

  double time_ms = measure_time(0, size_ - 1, data);

  if (rank_ == 0) {
    std::cout << "Long chain (" << size_ << " processes): " << time_ms << " ms for 8000 elements" << std::endl;

    EXPECT_LT(time_ms, 500.0);
  }
}

}  // namespace iskhakov_d_linear_topology
