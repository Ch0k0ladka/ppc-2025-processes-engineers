#include "iskhakov_d_graham_convex_hull/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_graham_convex_hull {

namespace {
constexpr double kEpsilon = 1e-9;
}  // namespace

IskhakovDGrahamConvexHullMPI::IskhakovDGrahamConvexHullMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<Point>();
}

bool IskhakovDGrahamConvexHullMPI::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool IskhakovDGrahamConvexHullMPI::PreProcessingImpl() {
  return true;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::GrahamScan(const std::vector<Point> &input_points) {
  std::vector<Point> points = input_points;
  if (points.size() < 3) {
    return points;
  }

  size_t index_min_point = 0;
  for (size_t index_vector = 1; index_vector < points.size(); index_vector++) {
    if (points[index_vector].y < points[index_min_point].y ||
        (std::abs(points[index_vector].y - points[index_min_point].y) < kEpsilon &&
         points[index_vector].x < points[index_min_point].x)) {
      index_min_point = index_vector;
    }
  }

  std::swap(points[0], points[index_min_point]);
  Point start_point = points[0];

  auto orientation = [](const Point &pivot, const Point &p1, const Point &p2) {
    return (p1.x - pivot.x) * (p2.y - pivot.y) - (p1.y - pivot.y) * (p2.x - pivot.x);
  };

  std::sort(points.begin() + 1, points.end(), [&start_point, &orientation](const Point &point1, const Point &point2) {
    return orientation(start_point, point1, point2) > 0;
  });

  std::vector<Point> hull;
  hull.push_back(points[0]);
  hull.push_back(points[1]);

  for (size_t index_vector = 2; index_vector < points.size(); index_vector++) {
    while (hull.size() >= 2 && orientation(hull[hull.size() - 2], hull.back(), points[index_vector]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[index_vector]);
  }

  return hull;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::MergeHulls(const std::vector<Point> &hull1,
                                                            const std::vector<Point> &hull2) {
  if (hull1.empty()) {
    return hull2;
  }
  if (hull2.empty()) {
    return hull1;
  }

  std::vector<Point> combined_hull;
  combined_hull.reserve(hull1.size() + hull2.size());
  combined_hull.insert(combined_hull.end(), hull1.begin(), hull1.end());
  combined_hull.insert(combined_hull.end(), hull2.begin(), hull2.end());

  std::sort(combined_hull.begin(), combined_hull.end(), [](const Point &left, const Point &right) {
    if (std::abs(left.x - right.x) < kEpsilon) {
      return left.y < right.y;
    }
    return left.x < right.x;
  });

  auto last = std::unique(combined_hull.begin(), combined_hull.end(), [](const Point &first, const Point &second) {
    return std::abs(first.x - second.x) < kEpsilon && std::abs(first.y - second.y) < kEpsilon;
  });
  combined_hull.erase(last, combined_hull.end());

  return GrahamScan(combined_hull);
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::PrepareAndDistributeData(int world_rank, int world_size) {
  std::vector<Point> all_points;
  int points_count = 0;

  if (world_rank == 0) {
    all_points = GetInput();
    points_count = static_cast<int>(all_points.size());
  }

  MPI_Bcast(&points_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (points_count < 3) {
    if (world_rank == 0) {
      return all_points;
    } else {
      return std::vector<Point>();
    }
  }

  int optimal_active_procs = world_size;

  while (optimal_active_procs > 1 && points_count < 3 * optimal_active_procs) {
    optimal_active_procs--;
  }

  if (optimal_active_procs == 0) {
    optimal_active_procs = 1;
  }

  bool is_active = (world_rank < optimal_active_procs);

  int active_procs = optimal_active_procs;
  int active_rank = is_active ? world_rank : -1;

  int base_points_count = points_count / active_procs;
  int remainder = points_count % active_procs;

  int my_count_points = 0;
  if (is_active) {
    my_count_points = base_points_count + (active_rank < remainder ? 1 : 0);
  }

  std::vector<int> proc_points_counts(world_size, 0);
  MPI_Gather(&my_count_points, 1, MPI_INT, proc_points_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displacements(world_size, 0);
  if (world_rank == 0) {
    int offset = 0;
    for (int i = 0; i < world_size; i++) {
      displacements[i] = offset;
      offset += proc_points_counts[i];
    }
  }

  MPI_Bcast(displacements.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> all_x, all_y;

  if (world_rank == 0) {
    all_x.resize(points_count);
    all_y.resize(points_count);

    for (int i = 0; i < points_count; i++) {
      all_x[i] = all_points[i].x;
      all_y[i] = all_points[i].y;
    }
  }

  std::vector<double> local_x(my_count_points);
  std::vector<double> local_y(my_count_points);

  MPI_Scatterv(all_x.data(), proc_points_counts.data(), displacements.data(), MPI_DOUBLE, local_x.data(),
               my_count_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(all_y.data(), proc_points_counts.data(), displacements.data(), MPI_DOUBLE, local_y.data(),
               my_count_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<Point> local_points;
  local_points.reserve(my_count_points);
  for (int i = 0; i < my_count_points; i++) {
    local_points.emplace_back(local_x[i], local_y[i]);
  }

  return local_points;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::BuildLocalHull([[maybe_unused]] int world_rank,
                                                                const std::vector<double> &local_x,
                                                                const std::vector<double> &local_y,
                                                                int my_point_count) {
  std::vector<Point> local_hull;
  if (my_point_count > 0) {
    std::vector<Point> local_points;
    local_points.reserve(my_point_count);

    for (int i = 0; i < my_point_count; i++) {
      local_points.emplace_back(local_x[i], local_y[i]);
    }

    if (local_points.size() >= 3) {
      local_hull = GrahamScan(local_points);
    } else {
      local_hull = local_points;
    }
  }
  return local_hull;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::MergeHullsBinaryTree(int world_rank, int world_size,
                                                                      const std::vector<Point> &local_hull) {
  std::vector<Point> current_hull = local_hull;

  int active_procs = 1;
  while (active_procs * 2 <= world_size) {
    active_procs *= 2;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (int step = 1; step < active_procs; step *= 2) {
    int partner = world_rank ^ step;

    if (partner >= world_size) {
      continue;
    }

    int my_hull_size = static_cast<int>(current_hull.size());
    int partner_hull_size = 0;

    MPI_Status status;
    MPI_Sendrecv(&my_hull_size, 1, MPI_INT, partner, 0, &partner_hull_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
                 &status);

    MPI_Barrier(MPI_COMM_WORLD);

    if (partner_hull_size > 0 && my_hull_size == 0) {
      std::vector<double> remote_x(partner_hull_size);
      std::vector<double> remote_y(partner_hull_size);

      MPI_Recv(remote_x.data(), partner_hull_size, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(remote_y.data(), partner_hull_size, MPI_DOUBLE, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<Point> remote_hull;
      remote_hull.reserve(partner_hull_size);
      for (int i = 0; i < partner_hull_size; i++) {
        remote_hull.emplace_back(remote_x[i], remote_y[i]);
      }
      current_hull = remote_hull;

    } else if (my_hull_size > 0 && partner_hull_size == 0) {
      std::vector<double> hull_x(my_hull_size);
      std::vector<double> hull_y(my_hull_size);
      for (int i = 0; i < my_hull_size; i++) {
        hull_x[i] = current_hull[i].x;
        hull_y[i] = current_hull[i].y;
      }

      MPI_Send(hull_x.data(), my_hull_size, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD);
      MPI_Send(hull_y.data(), my_hull_size, MPI_DOUBLE, partner, 2, MPI_COMM_WORLD);

    } else if (my_hull_size > 0 && partner_hull_size > 0) {
      std::vector<double> hull_x(my_hull_size);
      std::vector<double> hull_y(my_hull_size);
      for (int i = 0; i < my_hull_size; i++) {
        hull_x[i] = current_hull[i].x;
        hull_y[i] = current_hull[i].y;
      }

      std::vector<double> remote_x(partner_hull_size);
      std::vector<double> remote_y(partner_hull_size);

      MPI_Sendrecv(hull_x.data(), my_hull_size, MPI_DOUBLE, partner, 1, remote_x.data(), partner_hull_size, MPI_DOUBLE,
                   partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(hull_y.data(), my_hull_size, MPI_DOUBLE, partner, 2, remote_y.data(), partner_hull_size, MPI_DOUBLE,
                   partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<Point> remote_hull;
      remote_hull.reserve(partner_hull_size);
      for (int i = 0; i < partner_hull_size; i++) {
        remote_hull.emplace_back(remote_x[i], remote_y[i]);
      }

      current_hull = MergeHulls(current_hull, remote_hull);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (world_rank >= active_procs) {
    current_hull.clear();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return current_hull;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::BroadcastFinalResult(int world_rank, [[maybe_unused]] int world_size,
                                                                      const std::vector<Point> &final_hull_root) {
  int final_hull_size = 0;

  if (world_rank == 0) {
    final_hull_size = static_cast<int>(final_hull_root.size());
  }

  MPI_Bcast(&final_hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<Point> final_result;
  if (final_hull_size > 0) {
    if (world_rank == 0) {
      final_result = final_hull_root;

      std::vector<double> final_x(final_hull_size);
      std::vector<double> final_y(final_hull_size);

      for (int i = 0; i < final_hull_size; i++) {
        final_x[i] = final_result[i].x;
        final_y[i] = final_result[i].y;
      }

      MPI_Bcast(final_x.data(), final_hull_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(final_y.data(), final_hull_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
      std::vector<double> final_x(final_hull_size);
      std::vector<double> final_y(final_hull_size);

      MPI_Bcast(final_x.data(), final_hull_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(final_y.data(), final_hull_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      final_result.reserve(final_hull_size);
      for (int i = 0; i < final_hull_size; i++) {
        final_result.emplace_back(final_x[i], final_y[i]);
      }
    }
  }

  return final_result;
}

bool IskhakovDGrahamConvexHullMPI::RunImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<Point> local_points = PrepareAndDistributeData(world_rank, world_size);

  bool is_active = !local_points.empty();

  std::vector<Point> local_hull;
  if (is_active) {
    if (local_points.size() >= 3) {
      local_hull = GrahamScan(local_points);
    } else {
      local_hull = local_points;
    }
  }

  std::vector<Point> merged_hull = MergeHullsBinaryTree(world_rank, world_size, local_hull);

  std::vector<Point> final_result = BroadcastFinalResult(world_rank, world_size, merged_hull);

  GetOutput() = final_result;

  return true;
}

bool IskhakovDGrahamConvexHullMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_graham_convex_hull
