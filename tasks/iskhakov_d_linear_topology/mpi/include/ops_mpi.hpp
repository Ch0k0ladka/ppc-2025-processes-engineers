#pragma once

#include "iskhakov_d_linear_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace iskhakov_d_linear_topology {

class IskhakovDLinearTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit IskhakovDLinearTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace iskhakov_d_linear_topology
