#pragma once

#include <vector>

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

  static std::vector<Point> GrahamScan(const std::vector<Point> &input_points);
  static std::vector<Point> MergeHulls(const std::vector<Point> &hull_left, const std::vector<Point> &hull_right);

  static int CalculateOptimalActiveProcs(int points_count, int world_size);
  std::vector<Point> PrepareAndDistributeData(int world_rank, int world_size, int &optimal_active_procs_out);
  static std::vector<Point> MergeHullsBinaryTree(int world_rank, const std::vector<Point> &local_hull,
                                                 int optimal_active_procs);
  static std::vector<Point> BroadcastFinalResult(int world_rank, const std::vector<Point> &root_hull);
};

}  // namespace iskhakov_d_graham_convex_hull
