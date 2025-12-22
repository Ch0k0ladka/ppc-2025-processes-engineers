#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"

#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"

namespace iskhakov_d_linear_topology {

IskhakovDLinearTopologySEQ::IskhakovDLinearTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Message{};
}

bool IskhakovDLinearTopologySEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.head_process < 0) {
    return false;
  }

  if (input.tail_process < 0) {
    return false;
  }

  if (input.head_process != input.tail_process) {
    return false;
  }

  if (input.data.empty()) {
    return false;
  }

  if (input.delivered) {
    return false;
  }

  return true;
}

bool IskhakovDLinearTopologySEQ::PreProcessingImpl() {
  return true;
}

bool IskhakovDLinearTopologySEQ::RunImpl() {
  const auto &input = GetInput();

  Message result;
  result.head_process = input.head_process;
  result.tail_process = input.tail_process;
  result.SetData(input.data);
  result.delivered = true;

  GetOutput() = result;
  return true;
}

bool IskhakovDLinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_linear_topology
