#include "iskhakov_d_graham_convex_hull/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"

namespace iskhakov_d_graham_convex_hull {

namespace {

constexpr double kEpsilon = 1e-9;
constexpr int kMinPointsPerProcess = 10;

constexpr int kTagSize = 0;
constexpr int kTagX = 1;
constexpr int kTagY = 2;

int CalculateLocalOrSendcount(int index, int active_procs, int base_count, int remainder) {
  if (index < active_procs) {
    return base_count + (index < remainder ? 1 : 0);
  }
  return 0;
}

int ComputeOrientation(const Point &p, const Point &q, const Point &r) {
  const double value = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));

  if (std::abs(value) < kEpsilon) {
    return 0;
  }
  return (value > 0) ? 1 : 2;
}

double ComputeDistanceSquared(const Point &a, const Point &b) {
  return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}

std::size_t FindMinPointIndex(const std::vector<Point> &points) {
  std::size_t min_index = 0;
  for (std::size_t i = 1; i < points.size(); ++i) {
    if (points[i].y < points[min_index].y ||
        (std::abs(points[i].y - points[min_index].y) < kEpsilon && points[i].x < points[min_index].x)) {
      min_index = i;
    }
  }
  return min_index;
}

void FilterCollinearPoints(std::vector<Point> &points, const Point &pivot) {
  std::size_t unique_count = 1;
  for (std::size_t i = 1; i < points.size(); ++i) {
    while (i + 1 < points.size() && ComputeOrientation(pivot, points[i], points[i + 1]) == 0) {
      ++i;
    }
    points[unique_count++] = points[i];
  }
  points.resize(unique_count);
}

void BuildConvexHull(const std::vector<Point> &points, std::vector<Point> &hull) {
  hull.clear();
  if (points.size() < 3) {
    hull = points;
    return;
  }

  hull.reserve(points.size());
  hull.push_back(points[0]);
  hull.push_back(points[1]);
  hull.push_back(points[2]);

  for (std::size_t i = 3; i < points.size(); ++i) {
    while (hull.size() >= 2 && ComputeOrientation(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) != 2) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }
}

}  // namespace

IskhakovDGrahamConvexHullMPI::IskhakovDGrahamConvexHullMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
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

  std::vector<Point> points{input_points};

  const std::size_t min_index = FindMinPointIndex(points);
  std::swap(points[0], points[min_index]);
  const Point pivot = points[0];

  std::sort(points.begin() + 1, points.end(), [&pivot](const Point &a, const Point &b) {
    const int orient = ComputeOrientation(pivot, a, b);
    if (orient == 0) {
      const double dist_a = ComputeDistanceSquared(a, pivot);
      const double dist_b = ComputeDistanceSquared(b, pivot);
      return dist_a < dist_b;
    }
    return orient == 2;
  });

  FilterCollinearPoints(points, pivot);

  std::vector<Point> hull;
  BuildConvexHull(points, hull);

  return hull;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::MergeHulls(const std::vector<Point> &hull_left,
                                                            const std::vector<Point> &hull_right) {
  if (hull_left.empty()) {
    return hull_right;
  }
  if (hull_right.empty()) {
    return hull_left;
  }

  std::vector<Point> merged;
  merged.reserve(hull_left.size() + hull_right.size());
  merged.insert(merged.end(), hull_left.begin(), hull_left.end());
  merged.insert(merged.end(), hull_right.begin(), hull_right.end());

  return GrahamScan(merged);
}

int IskhakovDGrahamConvexHullMPI::CalculateOptimalActiveProcs(int points_count, int world_size) {
  int active_procs = world_size;

  while (active_procs > 1 && points_count / active_procs < kMinPointsPerProcess) {
    --active_procs;
  }

  while (active_procs > 1 && points_count / active_procs < 3) {
    --active_procs;
  }

  return std::max(active_procs, 1);
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::PrepareAndDistributeData(int world_rank, int world_size,
                                                                          int &active_procs_out) {
  int total_points = 0;
  if (world_rank == 0) {
    total_points = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int active_procs = CalculateOptimalActiveProcs(total_points, world_size);
  active_procs_out = active_procs;

  const bool is_active = world_rank < active_procs;

  const int base_count = total_points / active_procs;
  const int remainder = total_points % active_procs;

  const int local_count = is_active ? CalculateLocalOrSendcount(world_rank, active_procs, base_count, remainder) : 0;

  std::vector<int> sendcounts(world_size, 0);
  std::vector<int> displs(world_size, 0);

  if (world_rank == 0) {
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      sendcounts[i] = CalculateLocalOrSendcount(i, active_procs, base_count, remainder);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> all_x;
  std::vector<double> all_y;

  if (world_rank == 0) {
    all_x.resize(total_points);
    all_y.resize(total_points);
    for (int i = 0; i < total_points; ++i) {
      all_x[i] = GetInput()[static_cast<std::size_t>(i)].x;
      all_y[i] = GetInput()[static_cast<std::size_t>(i)].y;
    }
  }

  std::vector<double> local_x(static_cast<std::size_t>(local_count));
  std::vector<double> local_y(static_cast<std::size_t>(local_count));

  if (local_count > 0) {
    MPI_Scatterv(all_x.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_x.data(), local_count, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(all_y.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_y.data(), local_count, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }

  std::vector<Point> local_points;
  local_points.reserve(static_cast<std::size_t>(local_count));
  for (int i = 0; i < local_count; ++i) {
    local_points.emplace_back(local_x[i], local_y[i]);
  }

  return local_points;
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::MergeHullsBinaryTree(int world_rank,
                                                                      const std::vector<Point> &local_hull,
                                                                      int active_procs) {
  std::vector<Point> current_hull = local_hull;

  for (int step = 1; step < active_procs; step <<= 1) {
    const int partner = world_rank ^ step;

    if (world_rank < active_procs && partner < active_procs) {
      int my_size = static_cast<int>(current_hull.size());
      int partner_size = 0;

      MPI_Sendrecv(&my_size, 1, MPI_INT, partner, kTagSize, &partner_size, 1, MPI_INT, partner, kTagSize,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<double> my_x(my_size);
      std::vector<double> my_y(my_size);

      for (int i = 0; i < my_size; ++i) {
        my_x[i] = current_hull[static_cast<std::size_t>(i)].x;
        my_y[i] = current_hull[static_cast<std::size_t>(i)].y;
      }

      std::vector<double> remote_x(partner_size);
      std::vector<double> remote_y(partner_size);

      MPI_Sendrecv(my_x.data(), my_size, MPI_DOUBLE, partner, kTagX, remote_x.data(), partner_size, MPI_DOUBLE, partner,
                   kTagX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(my_y.data(), my_size, MPI_DOUBLE, partner, kTagY, remote_y.data(), partner_size, MPI_DOUBLE, partner,
                   kTagY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<Point> remote_hull;
      remote_hull.reserve(static_cast<std::size_t>(partner_size));
      for (int i = 0; i < partner_size; ++i) {
        remote_hull.emplace_back(remote_x[i], remote_y[i]);
      }

      current_hull = MergeHulls(current_hull, remote_hull);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  return (world_rank < active_procs) ? current_hull : std::vector<Point>{};
}

std::vector<Point> IskhakovDGrahamConvexHullMPI::BroadcastFinalResult(int world_rank,
                                                                      const std::vector<Point> &root_hull) {
  int hull_size = 0;
  if (world_rank == 0) {
    hull_size = static_cast<int>(root_hull.size());
  }

  MPI_Bcast(&hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<Point> result;
  result.reserve(static_cast<std::size_t>(hull_size));

  std::vector<double> x(static_cast<std::size_t>(hull_size));
  std::vector<double> y(static_cast<std::size_t>(hull_size));

  if (world_rank == 0) {
    for (int i = 0; i < hull_size; ++i) {
      x[i] = root_hull[static_cast<std::size_t>(i)].x;
      y[i] = root_hull[static_cast<std::size_t>(i)].y;
    }
  }

  MPI_Bcast(x.data(), hull_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(y.data(), hull_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int i = 0; i < hull_size; ++i) {
    result.emplace_back(x[i], y[i]);
  }

  return result;
}

bool IskhakovDGrahamConvexHullMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int points_count = 0;
  if (world_rank == 0) {
    points_count = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&points_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int active_procs = CalculateOptimalActiveProcs(points_count, world_size);

  if (active_procs == 1) {
    std::vector<Point> result;
    if (world_rank == 0) {
      result = GrahamScan(GetInput());
    }
    GetOutput() = BroadcastFinalResult(world_rank, result);
    return true;
  }

  int actual_active_procs = 0;
  const std::vector<Point> local_points = PrepareAndDistributeData(world_rank, world_size, actual_active_procs);

  std::vector<Point> local_hull;
  if (!local_points.empty()) {
    local_hull = GrahamScan(local_points);
  }

  const std::vector<Point> merged = MergeHullsBinaryTree(world_rank, local_hull, actual_active_procs);

  const std::vector<Point> root_hull = (world_rank == 0) ? merged : std::vector<Point>{};

  GetOutput() = BroadcastFinalResult(world_rank, root_hull);
  return true;
}

bool IskhakovDGrahamConvexHullMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_graham_convex_hull
