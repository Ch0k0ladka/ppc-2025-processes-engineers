#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <numeric>
#include <tuple>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_trapezoidal_integration {

IskhakovDTrapezoidalIntegrationMPI::IskhakovDTrapezoidalIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool IskhakovDTrapezoidalIntegrationMPI::ValidationImpl() {
  auto &input = GetInput();
  double lower_level = std::get<0>(input);
  double top_level = std::get<1>(input);
  int number_steps = std::get<3>(input);

  return (lower_level < top_level) && (number_steps > 0);
}

bool IskhakovDTrapezoidalIntegrationMPI::PreProcessingImpl() {
  return true;
}

bool IskhakovDTrapezoidalIntegrationMPI::RunImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto &input = GetInput();

  double lower_level = 0.0;
  double top_level = 0.0;
  int number_steps = 0;
  double local_sum = 0.0;

  if (world_rank == 0) {
    lower_level = std::get<0>(input);
    top_level = std::get<1>(input);
    number_steps = std::get<3>(input);
  }

  MPI_Bcast(&lower_level, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&top_level, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&number_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto input_function = std::get<2>(input);

  double step = (top_level - lower_level) / static_cast<double>(number_steps);

  if (world_rank == 0) {
    local_sum = (input_function(lower_level) + input_function(top_level)) / 2.0;
  }

  for (int step_index = world_rank + 1; step_index < number_steps; step_index += world_size) {
    local_sum += input_function(lower_level + step * step_index);
  }

  local_sum *= step;

  double result = 0.0;
  MPI_Allreduce(&local_sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = result;

  return true;
}

bool IskhakovDTrapezoidalIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_trapezoidal_integration
