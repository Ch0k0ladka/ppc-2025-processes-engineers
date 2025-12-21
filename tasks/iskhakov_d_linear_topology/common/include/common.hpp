#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace iskhakov_d_linear_topology {

struct Message {
  int head_process = 0;
  int tail_process = 0;
  int data_size = 0;
  std::vector<int> data;
  bool delivered = false;
};

struct Result {
  Message message;
  int process_count = 0;

  Result() : message(), process_count(0) {}

  Result(Message msg, int count) : message(std::move(msg)), process_count(count) {}
};

using InType = Message;
using OutType = Result;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace iskhakov_d_linear_topology
