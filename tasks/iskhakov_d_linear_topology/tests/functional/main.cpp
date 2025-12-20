#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"
#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace iskhakov_d_linear_topology {

class IskhakovDLinearTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[input, expected] = test_param;
    return "head_" + std::to_string(input.head_process) + "_tail_" + std::to_string(input.tail_process) + "_size_" +
           std::to_string(input.data.size());
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const auto &[input, expected] = params;

    input_data_ = input;
    expected_data_ = expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.data != expected_data_.data) {
      return false;
    }

    if (output_data.head_process != expected_data_.head_process) {
      return false;
    }

    if (output_data.tail_process != expected_data_.tail_process) {
      return false;
    }

    if (output_data.delivered != expected_data_.delivered) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_;
};

namespace {

std::vector<int> GenerateTestVector(size_t size, int start = 1) {
  std::vector<int> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<int>(i + start);
  }
  return data;
}

std::vector<int> GenerateConsecutiveVector(size_t size, int first = 10) {
  std::vector<int> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<int>(first + i);
  }
  return data;
}

std::vector<int> GenerateConstantVector(size_t size, int value = 42) {
  return std::vector<int>(size, value);
}

std::vector<int> GenerateIncreasingVector(size_t size, int step = 3) {
  std::vector<int> data(size);
  int value = 1;
  for (size_t i = 0; i < size; ++i) {
    data[i] = value;
    value += step;
  }
  return data;
}

TestType CreateTestData(int head, int tail, const std::vector<int> &data) {
  InType input{head, tail, data, false};
  OutType expected{head, tail, data, true};
  return std::make_tuple(input, expected);
}

const std::array<TestType, 27> kTestParam = {
    CreateTestData(0, 1, GenerateTestVector(3)),
    CreateTestData(0, 1, GenerateTestVector(5)),
    CreateTestData(1, 0, GenerateTestVector(4)),
    CreateTestData(1, 0, GenerateTestVector(2)),

    CreateTestData(0, 2, GenerateTestVector(3, 10)),
    CreateTestData(2, 0, GenerateTestVector(4, 20)),
    CreateTestData(0, 2, GenerateConsecutiveVector(5, 100)),
    CreateTestData(2, 0, GenerateConsecutiveVector(6, 200)),

    CreateTestData(0, 3, GenerateTestVector(2, 30)),
    CreateTestData(0, 3, GenerateConstantVector(4, 99)),
    CreateTestData(3, 0, GenerateConstantVector(5, 77)),

    CreateTestData(1, 3, GenerateTestVector(3, 50)),
    CreateTestData(3, 1, GenerateTestVector(4, 60)),
    CreateTestData(1, 3, GenerateIncreasingVector(5, 5)),
    CreateTestData(3, 1, GenerateIncreasingVector(6, 7)),

    CreateTestData(1, 2, GenerateTestVector(3, 70)),
    CreateTestData(2, 1, GenerateTestVector(4, 80)),
    CreateTestData(1, 2, std::vector<int>{1, 3, 5, 7, 9}),
    CreateTestData(2, 1, std::vector<int>{2, 4, 6, 8, 10}),

    CreateTestData(0, 3, GenerateTestVector(100, 1)),
    CreateTestData(3, 0, GenerateTestVector(50, 1000)),

    CreateTestData(0, 3, std::vector<int>{999}),
    CreateTestData(2, 1, std::vector<int>{-1}),

    CreateTestData(0, 2, std::vector<int>{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}),

    CreateTestData(1, 3, std::vector<int>{15, 8, 42, 23, 17, 9, 31}),

    CreateTestData(0, 3, std::vector<int>{7, 7, 7, 7, 7}),
    CreateTestData(3, 0, std::vector<int>{3, 3, 3})};

TEST_P(IskhakovDLinearTopologyFuncTests, LinearTopology) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IskhakovDLinearTopologyMPI, InType>(kTestParam, PPC_SETTINGS_iskhakov_d_linear_topology),
    ppc::util::AddFuncTask<IskhakovDLinearTopologySEQ, InType>(kTestParam, PPC_SETTINGS_iskhakov_d_linear_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IskhakovDLinearTopologyFuncTests::PrintFuncTestName<IskhakovDLinearTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(LinearTopologyTests, IskhakovDLinearTopologyFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace iskhakov_d_linear_topology
