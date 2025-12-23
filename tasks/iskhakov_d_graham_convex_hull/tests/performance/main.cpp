#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "iskhakov_d_graham_convex_hull/common/include/common.hpp"
#include "iskhakov_d_graham_convex_hull/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_graham_convex_hull/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace iskhakov_d_graham_convex_hull {

namespace {
constexpr std::size_t kPointCount = 1000000;
constexpr int kCoordinateMin = 0;
constexpr int kCoordinateMax = 10000;
constexpr int kMaxGenerationAttempts = 100;
constexpr double kBoundEpsilon = 1e-6;

struct PairHash {
  std::size_t operator()(const std::pair<int, int> &p) const noexcept {
    return static_cast<std::size_t>(p.first) ^ (static_cast<std::size_t>(p.second) << 16);
  }
};
}  // namespace

class IskhakovDGrahamConvexHullPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(kCoordinateMin, kCoordinateMax);

    std::vector<Point> points;
    points.reserve(kPointCount);

    std::unordered_set<std::pair<int, int>, PairHash> unique_points;
    unique_points.reserve(kPointCount);

    for (std::size_t i = 0; i < kPointCount; ++i) {
      int attempts = 0;
      bool point_added = false;

      while (attempts < kMaxGenerationAttempts && !point_added) {
        const int x = dist(gen);
        const int y = dist(gen);

        const auto [iter, inserted] = unique_points.emplace(x, y);
        if (inserted) {
          points.emplace_back(static_cast<double>(x), static_cast<double>(y));
          point_added = true;
        }
        ++attempts;
      }

      if (!point_added) {
        const int x = dist(gen);
        const int y = dist(gen);
        points.emplace_back(static_cast<double>(x), static_cast<double>(y));
      }
    }

    input_data_ = std::move(points);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() > kPointCount) {
      return false;
    }

    for (const auto &point : output_data) {
      if (point.x < kCoordinateMin - kBoundEpsilon || point.x > kCoordinateMax + kBoundEpsilon ||
          point.y < kCoordinateMin - kBoundEpsilon || point.y > kCoordinateMax + kBoundEpsilon) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(IskhakovDGrahamConvexHullPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, IskhakovDGrahamConvexHullMPI, IskhakovDGrahamConvexHullSEQ>(
        PPC_SETTINGS_iskhakov_d_graham_convex_hull);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IskhakovDGrahamConvexHullPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IskhakovDGrahamConvexHullPerfsTests, kGtestValues, kPerfTestName);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IskhakovDGrahamConvexHullPerfTests);

}  // namespace iskhakov_d_graham_convex_hull
