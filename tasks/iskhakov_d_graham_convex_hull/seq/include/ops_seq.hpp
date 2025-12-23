#pragma once

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"
#include "task/include/task.hpp"

namespace iskhakov_d_graham_convex_hull {

class IskhakovDGrahamConvexHullSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IskhakovDGrahamConvexHullSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace  iskhakov_d_graham_convex_hull
