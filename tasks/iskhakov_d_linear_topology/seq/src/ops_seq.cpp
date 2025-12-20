#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"

#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"

namespace iskhakov_d_linear_topology {

IskhakovDLinearTopologySEQ::IskhakovDLinearTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool IskhakovDLinearTopologySEQ::ValidationImpl() {
  const auto &input = GetInput();

  if ((!input.data.empty()) && (!input.delivered)) {
    return true;
  }

  return false;
}

bool IskhakovDLinearTopologySEQ::PreProcessingImpl() {
  return true;
}

bool IskhakovDLinearTopologySEQ::RunImpl() {
  const auto &input = GetInput();

  std::vector<int> local_data = input.data;
  bool delivered = true;

  GetOutput() = {input.head_process, input.tail_process, local_data, delivered};

  return true;
}

bool IskhakovDLinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_linear_topology
