#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <numeric>

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

  return (input.lower_level < input.top_level) && (input.number_steps > 0);
}

bool IskhakovDTrapezoidalIntegrationMPI::PreProcessingImpl() {
  return true;
}

bool IskhakovDTrapezoidalIntegrationMPI::RunImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  double lower_level = 0.0, top_level = 0.0;
  int number_steps = 0;
  double local_sum = 0.0;

  auto &input = GetInput();

  if (world_rank == 0) {
    lower_level = input.lower_level;
    top_level = input.top_level;
    number_steps = input.number_steps;
  }

  MPI_Bcast(&lower_level, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&top_level, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&number_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto &input_function = input.function;
  double step = (top_level - lower_level) / number_steps;

  if (world_rank == 0) {
    local_sum = (input_function(lower_level) + input_function(top_level)) / 2.0;
  }

  for (int i = world_rank + 1; i < number_steps; i += world_size) {
    local_sum += input_function(lower_level + step * i);
  }

  local_sum *= step;

  double result;
  MPI_Allreduce(&local_sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = result;

  return true;
}

bool IskhakovDTrapezoidalIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_trapezoidal_integration
