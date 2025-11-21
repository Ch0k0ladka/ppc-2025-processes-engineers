#include "iskhakov_d_trapezoidal_integration/seq/include/ops_seq.hpp"

#include <cmath>
#include <numeric>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_trapezoidal_integration {

IskhakovDTrapezoidalIntegrationSEQ::IskhakovDTrapezoidalIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool IskhakovDTrapezoidalIntegrationSEQ::ValidationImpl() {
  auto &input = GetInput();

  return (input.lower_level < input.top_level) && (input.number_steps > 0);
}

bool IskhakovDTrapezoidalIntegrationSEQ::PreProcessingImpl() {
  return GetOutput() == 0.0;
}

bool IskhakovDTrapezoidalIntegrationSEQ::RunImpl() {
  auto &input = GetInput();

  double lower_level = input.lower_level;
  double top_level = input.top_level;
  auto &input_function = input.function;
  int number_steps = input.number_steps;

  double result = 0.0;
  double step = (top_level - lower_level) / number_steps;

  result = (input_function(lower_level) + input_function(top_level)) / 2.0;

  for (int i = 1; i < number_steps; i++) {
    result += input_function(lower_level + step * i);
  }

  result *= step;
  GetOutput() = result;

  return true;
}

bool IskhakovDTrapezoidalIntegrationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_trapezoidal_integration
