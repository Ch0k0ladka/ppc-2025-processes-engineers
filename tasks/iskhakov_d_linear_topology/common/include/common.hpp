#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace iskhakov_d_linear_topology {

struct Message {
  int head_process = 0;
  int tail_process = 0;
  bool delivered = false;
  std::vector<int> data;

  int data_size() const {
    return static_cast<int>(data.size());
  }

  void set_data(const std::vector<int> &new_data) {
    data = new_data;
  }

  void set_data(std::vector<int> &&new_data) {
    data = std::move(new_data);
  }
};

using InType = Message;
using OutType = Message;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace iskhakov_d_linear_topology
