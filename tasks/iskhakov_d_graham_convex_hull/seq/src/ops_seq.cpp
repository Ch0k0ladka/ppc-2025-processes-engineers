#include "iskhakov_d_graham_convex_hull/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"

namespace iskhakov_d_graham_convex_hull {

namespace {

constexpr double kEpsilon = 1e-9;

double ComputeOrientation(const Point &p, const Point &q, const Point &r) {
  return ((q.x - p.x) * (r.y - p.y)) - ((q.y - p.y) * (r.x - p.x));
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

std::vector<Point> FilterCollinearPoints(const std::vector<Point> &points, const Point &pivot) {
  std::vector<Point> filtered;
  filtered.reserve(points.size());
  filtered.push_back(points[0]);

  for (std::size_t i = 1; i < points.size(); ++i) {
    while (i + 1 < points.size() && std::abs(ComputeOrientation(pivot, points[i], points[i + 1])) < kEpsilon) {
      ++i;
    }
    filtered.push_back(points[i]);
  }

  return filtered;
}

std::vector<Point> BuildConvexHull(const std::vector<Point> &filtered) {
  if (filtered.size() < 3) {
    return filtered;
  }

  std::vector<Point> hull;
  hull.reserve(filtered.size());
  hull.push_back(filtered[0]);
  hull.push_back(filtered[1]);

  for (std::size_t i = 2; i < filtered.size(); ++i) {
    while (hull.size() >= 2) {
      const std::size_t last = hull.size() - 1;
      const double orient = ComputeOrientation(hull[last - 1], hull[last], filtered[i]);

      if (orient > kEpsilon) {
        break;
      }

      hull.pop_back();
    }
    hull.push_back(filtered[i]);
  }

  return hull;
}

}  // namespace

IskhakovDGrahamConvexHullSEQ::IskhakovDGrahamConvexHullSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool IskhakovDGrahamConvexHullSEQ::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool IskhakovDGrahamConvexHullSEQ::PreProcessingImpl() {
  return true;
}

bool IskhakovDGrahamConvexHullSEQ::RunImpl() {
  const std::vector<Point> &input_points = GetInput();

  if (input_points.size() < 3) {
    GetOutput() = input_points;
    return true;
  }

  std::vector<Point> points{input_points};
  const std::size_t min_index = FindMinPointIndex(points);
  std::swap(points[0], points[min_index]);
  const Point pivot = points[0];

  std::sort(points.begin() + 1, points.end(), [&pivot](const Point &a, const Point &b) {
    const double orient = ComputeOrientation(pivot, a, b);
    if (std::abs(orient) < kEpsilon) {
      const double dist_a = ComputeDistanceSquared(a, pivot);
      const double dist_b = ComputeDistanceSquared(b, pivot);
      return dist_a < dist_b;
    }
    return orient > 0.0;
  });

  const std::vector<Point> filtered = FilterCollinearPoints(points, pivot);

  GetOutput() = BuildConvexHull(filtered);
  return true;
}

bool IskhakovDGrahamConvexHullSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_graham_convex_hull
