#include "iskhakov_d_graham_convex_hull/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_graham_convex_hull {

namespace {
constexpr double kEpsilon = 1e-9;
}  // namespace

IskhakovDGrahamConvexHullSEQ::IskhakovDGrahamConvexHullSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<Point>();
}

bool IskhakovDGrahamConvexHullSEQ::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool IskhakovDGrahamConvexHullSEQ::PreProcessingImpl() {
  return true;
}

bool IskhakovDGrahamConvexHullSEQ::RunImpl() {
  std::vector<Point> points = GetInput();
  if (points.size() < 3) {
    GetOutput() = points;
    return true;
  }

  size_t index_min_point = 0;
  for (size_t index_vector = 1; index_vector < points.size(); index_vector++) {
    if (points[index_vector].y < points[index_min_point].y - kEpsilon ||
        (std::abs(points[index_vector].y - points[index_min_point].y) < kEpsilon &&
         points[index_vector].x < points[index_min_point].x - kEpsilon)) {
      index_min_point = index_vector;
    }
  }

  std::swap(points[0], points[index_min_point]);
  const Point &start_point = points[0];

  auto orientation = [](const Point &pivot, const Point &p1, const Point &p2) {
    return (p1.x - pivot.x) * (p2.y - pivot.y) - (p1.y - pivot.y) * (p2.x - pivot.x);
  };

  std::sort(points.begin() + 1, points.end(), [&start_point, &orientation](const Point &point1, const Point &point2) {
    double orient = orientation(start_point, point1, point2);
    if (std::abs(orient) < kEpsilon) {
      double dist1 = (point1.x - start_point.x) * (point1.x - start_point.x) +
                     (point1.y - start_point.y) * (point1.y - start_point.y);
      double dist2 = (point2.x - start_point.x) * (point2.x - start_point.x) +
                     (point2.y - start_point.y) * (point2.y - start_point.y);
      return dist1 < dist2;
    }
    return orient > 0;
  });

  if (points.size() > 2) {
    std::vector<Point> filtered_points;
    filtered_points.push_back(points[0]);

    for (size_t i = 1; i < points.size(); i++) {
      while (i + 1 < points.size() && std::abs(orientation(start_point, points[i], points[i + 1])) < kEpsilon) {
        i++;
      }
      filtered_points.push_back(points[i]);
    }
    points = std::move(filtered_points);
  }

  if (points.size() < 3) {
    GetOutput() = points;
    return true;
  }

  std::vector<Point> hull;
  hull.reserve(points.size());
  hull.push_back(points[0]);
  hull.push_back(points[1]);

  for (size_t index_vector = 2; index_vector < points.size(); index_vector++) {
    while (hull.size() >= 2) {
      double orient = orientation(hull[hull.size() - 2], hull.back(), points[index_vector]);
      if (orient > kEpsilon) {
        break;
      } else if (orient < -kEpsilon) {
        hull.pop_back();
      } else {
        double dist_last = (hull.back().x - hull[hull.size() - 2].x) * (hull.back().x - hull[hull.size() - 2].x) +
                           (hull.back().y - hull[hull.size() - 2].y) * (hull.back().y - hull[hull.size() - 2].y);
        double dist_current =
            (points[index_vector].x - hull[hull.size() - 2].x) * (points[index_vector].x - hull[hull.size() - 2].x) +
            (points[index_vector].y - hull[hull.size() - 2].y) * (points[index_vector].y - hull[hull.size() - 2].y);
        if (dist_current > dist_last) {
          hull.pop_back();
        } else {
          break;
        }
      }
    }
    hull.push_back(points[index_vector]);
  }

  if (hull.size() < 3) {
    GetOutput() = points;
  } else {
    GetOutput() = hull;
  }

  return true;
}

bool IskhakovDGrahamConvexHullSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_graham_convex_hull
