#include "iskhakov_d_graham_convex_hull/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_graham_convex_hull {

namespace {
constexpr double kEpsilon = 1e-9;
constexpr int MIN_POINTS_PER_PROCESS = 10;
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
  if (input_points.size() < 3) {
    return input_points;
  }

  std::vector<Point> points = input_points;

  size_t index_min_point = 0;
  for (size_t i = 1; i < points.size(); i++) {
    if (points[i].y < points[index_min_point].y ||
        (std::abs(points[i].y - points[index_min_point].y) < kEpsilon && points[i].x < points[index_min_point].x)) {
      index_min_point = i;
    }
  }

  std::swap(points[0], points[index_min_point]);
  Point pivot = points[0];

  auto orientation = [](const Point &p, const Point &q, const Point &r) {
    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (std::abs(val) < kEpsilon) {
      return 0;
    }
    return (val > 0) ? 1 : 2;
  };

  std::sort(points.begin() + 1, points.end(), [&pivot, &orientation](const Point &a, const Point &b) {
    int o = orientation(pivot, a, b);
    if (o == 0) {
      double dist1 = (a.x - pivot.x) * (a.x - pivot.x) + (a.y - pivot.y) * (a.y - pivot.y);
      double dist2 = (b.x - pivot.x) * (b.x - pivot.x) + (b.y - pivot.y) * (b.y - pivot.y);
      return dist1 < dist2;
    }
    return o == 2;
  });

  size_t m = 1;
  for (size_t i = 1; i < points.size(); i++) {
    while (i < points.size() - 1 && orientation(pivot, points[i], points[i + 1]) == 0) {
      i++;
    }
    points[m] = points[i];
    m++;
  }
  points.resize(m);

  if (points.size() < 3) {
    return points;
  }

  std::vector<Point> hull;
  hull.push_back(points[0]);
  hull.push_back(points[1]);
  hull.push_back(points[2]);

  for (size_t i = 3; i < points.size(); i++) {
    while (hull.size() >= 2 && orientation(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) != 2) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
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

  std::vector<Point> combined_points;
  combined_points.reserve(hull1.size() + hull2.size());
  combined_points.insert(combined_points.end(), hull1.begin(), hull1.end());
  combined_points.insert(combined_points.end(), hull2.begin(), hull2.end());

  return GrahamScan(combined_points);
}

int IskhakovDGrahamConvexHullMPI::CalculateOptimalActiveProcs(int points_count, int world_size) {
  if (points_count < 3) {
    return 1;
  }

  int optimal_active_procs = world_size;

  while (optimal_active_procs > 1 && points_count / optimal_active_procs < MIN_POINTS_PER_PROCESS) {
    optimal_active_procs--;
  }

  while (optimal_active_procs > 1 && points_count / optimal_active_procs < 3) {
    optimal_active_procs--;
  }

  if (optimal_active_procs == 0) {
    optimal_active_procs = 1;
  }

  return optimal_active_procs;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::PrepareAndDistributeData(int world_rank, int world_size,
                                                                          int &optimal_active_procs_out) {
  int points_count = 0;

  if (world_rank == 0) {
    points_count = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&points_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int optimal_active_procs = CalculateOptimalActiveProcs(points_count, world_size);
  optimal_active_procs_out = optimal_active_procs;

  bool is_active = (world_rank < optimal_active_procs);

  int base_points_count = points_count / optimal_active_procs;
  int remainder = points_count % optimal_active_procs;

  int my_points_count = is_active ? (base_points_count + (world_rank < remainder ? 1 : 0)) : 0;

  std::vector<int> sendcounts(world_size, 0);
  std::vector<int> displacements(world_size, 0);

  if (world_rank == 0) {
    int offset = 0;
    for (int i = 0; i < world_size; i++) {
      if (i < optimal_active_procs) {
        int proc_count = base_points_count + (i < remainder ? 1 : 0);
        sendcounts[i] = proc_count;
      } else {
        sendcounts[i] = 0;
      }
      displacements[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (is_active && my_points_count != sendcounts[world_rank]) {
    my_points_count = sendcounts[world_rank];
  }

  std::vector<double> all_x, all_y;
  if (world_rank == 0) {
    all_x.resize(points_count);
    all_y.resize(points_count);
    for (int i = 0; i < points_count; i++) {
      all_x[i] = GetInput()[i].x;
      all_y[i] = GetInput()[i].y;
    }
  }

  std::vector<double> local_x(my_points_count);
  std::vector<double> local_y(my_points_count);

  if (my_points_count > 0) {
    MPI_Scatterv(all_x.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE, local_x.data(), my_points_count,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_y.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE, local_y.data(), my_points_count,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  std::vector<Point> local_points;
  local_points.reserve(my_points_count);
  for (int i = 0; i < my_points_count; i++) {
    local_points.emplace_back(local_x[i], local_y[i]);
  }

  return local_points;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::MergeHullsBinaryTree(int world_rank,
                                                                      const std::vector<Point> &local_hull,
                                                                      int optimal_active_procs) {
  std::vector<Point> current_hull = local_hull;

  for (int step = 1; step < optimal_active_procs; step *= 2) {
    int partner = world_rank ^ step;

    if (world_rank < optimal_active_procs && partner < optimal_active_procs) {
      int my_hull_size = static_cast<int>(current_hull.size());
      int partner_hull_size;

      MPI_Sendrecv(&my_hull_size, 1, MPI_INT, partner, 0, &partner_hull_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

      if (my_hull_size > 0 && partner_hull_size == 0) {
        std::vector<double> hull_x(my_hull_size);
        std::vector<double> hull_y(my_hull_size);
        for (int i = 0; i < my_hull_size; i++) {
          hull_x[i] = current_hull[i].x;
          hull_y[i] = current_hull[i].y;
        }

        MPI_Send(hull_x.data(), my_hull_size, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD);
        MPI_Send(hull_y.data(), my_hull_size, MPI_DOUBLE, partner, 2, MPI_COMM_WORLD);
      } else if (my_hull_size == 0 && partner_hull_size > 0) {
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
      } else if (my_hull_size > 0 && partner_hull_size > 0) {
        std::vector<double> hull_x(my_hull_size);
        std::vector<double> hull_y(my_hull_size);
        for (int i = 0; i < my_hull_size; i++) {
          hull_x[i] = current_hull[i].x;
          hull_y[i] = current_hull[i].y;
        }

        std::vector<double> remote_x(partner_hull_size);
        std::vector<double> remote_y(partner_hull_size);

        MPI_Sendrecv(hull_x.data(), my_hull_size, MPI_DOUBLE, partner, 1, remote_x.data(), partner_hull_size,
                     MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(hull_y.data(), my_hull_size, MPI_DOUBLE, partner, 2, remote_y.data(), partner_hull_size,
                     MPI_DOUBLE, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<Point> remote_hull;
        remote_hull.reserve(partner_hull_size);
        for (int i = 0; i < partner_hull_size; i++) {
          remote_hull.emplace_back(remote_x[i], remote_y[i]);
        }

        current_hull = MergeHulls(current_hull, remote_hull);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  return (world_rank < optimal_active_procs) ? current_hull : std::vector<Point>();
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::BroadcastFinalResult(int world_rank,
                                                                      const std::vector<Point> &final_hull_root) {
  int final_hull_size = 0;

  if (world_rank == 0) {
    final_hull_size = static_cast<int>(final_hull_root.size());
  }

  MPI_Bcast(&final_hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (final_hull_size == 0) {
    return std::vector<Point>();
  }

  std::vector<Point> final_result;

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

  return final_result;
}

bool IskhakovDGrahamConvexHullMPI::RunImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int points_count = 0;
  if (world_rank == 0) {
    points_count = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&points_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (points_count < 3) {
    std::vector<Point> result;
    if (world_rank == 0) {
      result = GetInput();
    }
    GetOutput() = BroadcastFinalResult(world_rank, result);
    return true;
  }

  int optimal_active_procs = CalculateOptimalActiveProcs(points_count, world_size);

  if (optimal_active_procs == 1) {
    std::vector<Point> result;
    if (world_rank == 0) {
      result = GrahamScan(GetInput());
    }
    GetOutput() = BroadcastFinalResult(world_rank, result);
    return true;
  }

  int actual_optimal_procs;
  std::vector<Point> local_points = PrepareAndDistributeData(world_rank, world_size, actual_optimal_procs);

  std::vector<Point> local_hull;
  if (!local_points.empty()) {
    local_hull = GrahamScan(local_points);
  }

  std::vector<Point> merged_hull = MergeHullsBinaryTree(world_rank, local_hull, actual_optimal_procs);

  std::vector<Point> final_hull_root;
  if (world_rank == 0) {
    final_hull_root = merged_hull;
  }

  GetOutput() = BroadcastFinalResult(world_rank, final_hull_root);

  return true;
}

bool IskhakovDGrahamConvexHullMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_graham_convex_hull
