#pragma once

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"
#include "task/include/task.hpp"

namespace iskhakov_d_graham_convex_hull {

class IskhakovDGrahamConvexHullMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit IskhakovDGrahamConvexHullMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<Point> GrahamScan(const std::vector<Point> &input_points);
  std::vector<Point> MergeHulls(const std::vector<Point> &hull1, const std::vector<Point> &hull2);
  
  int CalculateOptimalActiveProcs(int points_count, int world_size);
  std::vector<Point> PrepareAndDistributeData(int world_rank, int world_size, int &optimal_active_procs_out);
  std::vector<Point> MergeHullsBinaryTree(int world_rank, const std::vector<Point> &local_hull, int optimal_active_procs);
  std::vector<Point> BroadcastFinalResult(int world_rank, const std::vector<Point> &final_hull_root);
};

}  // namespace  iskhakov_d_graham_convex_hull