#include "iskhakov_d_linear_topology/seq/include/ops_seq.hpp"

#include <tuple>
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

  int head_process = input.head_process;
  int tail_process = input.tail_process;

  std::vector<int> local_data(input.data.begin(), input.data.end());
  
  bool delivered = true;

  Message result;
  result.head_process = head_process;
  result.tail_process = tail_process;
  result.data = local_data; 
  result.delivered = delivered;

  GetOutput() = std::make_tuple(result, 1);

  return true;
}

bool IskhakovDLinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_linear_topology
