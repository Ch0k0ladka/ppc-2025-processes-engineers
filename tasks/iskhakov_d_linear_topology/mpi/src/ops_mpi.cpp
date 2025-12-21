#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <utility>
#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"

namespace iskhakov_d_linear_topology {

IskhakovDLinearTopologyMPI::IskhakovDLinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool IskhakovDLinearTopologyMPI::ValidationImpl() {
  const auto &input = GetInput();

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (input.head_process < 0) {
    return false;
  }
  if (input.head_process >= world_size) {
    return false;
  }

  if (input.tail_process < 0) {
    return false;
  }
  if (input.tail_process >= world_size) {
    return false;
  }

  int is_valid_local = 1;

  if (world_rank == input.head_process) {
    if (input.data.empty()) {
      is_valid_local = 0;
    }
    if (input.delivered) {
      is_valid_local = 0;
    }
  }

  int is_valid_global;
  MPI_Allreduce(&is_valid_local, &is_valid_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  return (is_valid_global == 1);
}

bool IskhakovDLinearTopologyMPI::PreProcessingImpl() {
  return true;
}

bool IskhakovDLinearTopologyMPI::RunImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  const auto &input = GetInput();

  int head_process = input.head_process;
  int tail_process = input.tail_process;

  Message result;
  result.head_process = head_process;
  result.tail_process = tail_process;

  if (head_process == tail_process) {
    if (world_rank == head_process) {
      result.data = input.data;
      result.delivered = true;
    } else {
      result.data = {};
      result.delivered = false;
    }
    GetOutput() = std::make_tuple(result, world_size);
    return true;
  }

  int direction;

  if (head_process < tail_process) {
    direction = 1;
  } else {
    direction = -1;
  }

  bool participate;
  if (direction > 0) {
    participate = ((world_rank >= head_process) && (world_rank <= tail_process));
  } else {
    participate = ((world_rank <= head_process) && (world_rank >= tail_process));
  }

  if (!participate) {
    result.data = {};
    result.delivered = false;
    GetOutput() = std::make_tuple(result, world_size);
    return true;
  }

  bool is_head = (world_rank == head_process);
  bool is_tail = (world_rank == tail_process);

  int previous_process = MPI_PROC_NULL;
  int next_process = MPI_PROC_NULL;

  if (!is_head) {
    previous_process = world_rank - direction;
  }

  if (!is_tail) {
    next_process = world_rank + direction;
  }

  std::vector<int> local_data;

  if (is_head) {
    local_data = input.data;

    int local_data_size = static_cast<int>(local_data.size());
    MPI_Send(&local_data_size, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
    MPI_Send(local_data.data(), local_data_size, MPI_INT, next_process, 1, MPI_COMM_WORLD);

    result.data = local_data;
    result.delivered = true;
  } else if (is_tail) {
    int local_data_size = 0;
    MPI_Recv(&local_data_size, 1, MPI_INT, previous_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_data.resize(local_data_size);
    MPI_Recv(local_data.data(), local_data_size, MPI_INT, previous_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    result.data = std::move(local_data);
    result.delivered = true;
  } else {
    int local_data_size = 0;
    MPI_Recv(&local_data_size, 1, MPI_INT, previous_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_data.resize(local_data_size);
    MPI_Recv(local_data.data(), local_data_size, MPI_INT, previous_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Send(&local_data_size, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
    MPI_Send(local_data.data(), local_data_size, MPI_INT, next_process, 1, MPI_COMM_WORLD);

    result.data = {};
    result.delivered = false;
  }

  GetOutput() = std::make_tuple(result, world_size);
  return true;
}

bool IskhakovDLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_linear_topology
