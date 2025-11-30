#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"

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

  int local_count = number_steps / world_size;
  if (world_rank < number_steps % world_size) {
    local_count++;
  }

  std::vector<int> elements_per_proc(world_size);
  std::vector<int> displacement(world_size);
  std::vector<double> points(number_steps + 1);

  if (world_rank == 0) {
    for (int i = 0; i <= number_steps; i++) {
      points[i] = lower_level + i * step;
    }

    int steps_per_process = number_steps / world_size;
    int remainder = number_steps % world_size;
    int current_displacement = 0;

    for (int i = 0; i < world_size; i++) {
      if (i < remainder) {
        elements_per_proc[i] = steps_per_process + 1;
      } else {
        elements_per_proc[i] = steps_per_process;
      }
      displacement[i] = current_displacement;
      current_displacement += elements_per_proc[i];
    }
  }

  std::vector<double> local_points(local_count);

  if (world_rank == 0) {
    MPI_Scatterv(points.data(), elements_per_proc.data(), displacement.data(), MPI_DOUBLE, local_points.data(),
                 local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, local_points.data(), local_count, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }

  double local_sum = 0.0;
  for (double point : local_points) {
    local_sum += input_function(point);
  }

  if (world_rank == 0) {
    local_sum -= input_function(local_points[0]) * 0.5;
  }
  if (world_rank == world_size - 1) {
    local_sum -= input_function(local_points.back()) * 0.5;
  }

  double result = 0.0;
  MPI_Allreduce(&local_sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = result * step;

  return true;
}

bool IskhakovDTrapezoidalIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_trapezoidal_integration
