#include "iskhakov_d_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <utility>
#include <vector>

#include "iskhakov_d_linear_topology/common/include/common.hpp"

namespace iskhakov_d_linear_topology {

IskhakovDLinearTopologyMPI::IskhakovDLinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Message{};
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

  int is_valid_global = 0;
  MPI_Allreduce(&is_valid_local, &is_valid_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  return (is_valid_global == 1);
}

bool IskhakovDLinearTopologyMPI::PreProcessingImpl() {
  return true;
}

static void SendData(int local_data_size, const std::vector<int> &local_data, int next_process) {
  std::array<MPI_Request, 2> requests;
  MPI_Isend(&local_data_size, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD, &requests[0]);
  MPI_Isend(local_data.data(), local_data_size, MPI_INT, next_process, 1, MPI_COMM_WORLD, &requests[1]);
  MPI_Waitall(2, requests.data(), MPI_STATUSES_IGNORE);
}

static void ReceiveData(int &local_data_size, std::vector<int> &local_data, int previous_process) {
  MPI_Recv(&local_data_size, 1, MPI_INT, previous_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  local_data.resize(local_data_size);
  MPI_Recv(local_data.data(), local_data_size, MPI_INT, previous_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
      result.SetData(input.data);
      result.delivered = true;
    } else {
      result.SetData({});
      result.delivered = false;
    }
    GetOutput() = result;
    return true;
  }

  int direction = 0;
  if (head_process < tail_process) {
    direction = 1;
  } else {
    direction = -1;
  }

  bool participate = false;
  if (direction > 0) {
    participate = ((world_rank >= head_process) && (world_rank <= tail_process));
  } else {
    participate = ((world_rank <= head_process) && (world_rank >= tail_process));
  }

  if (!participate) {
    result.SetData({});
    result.delivered = false;
    GetOutput() = result;
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
  int local_data_size = 0;

  if (is_head) {
    local_data = input.data;
    local_data_size = static_cast<int>(local_data.size());

    SendData(local_data_size, local_data, next_process);

    result.SetData({});
    result.delivered = false;
  } else if (is_tail) {
    ReceiveData(local_data_size, local_data, previous_process);

    result.SetData(std::move(local_data));
    result.delivered = true;
  } else {
    ReceiveData(local_data_size, local_data, previous_process);
    SendData(local_data_size, local_data, next_process);

    result.SetData({});
    result.delivered = false;
  }

  GetOutput() = result;
  return true;
}

bool IskhakovDLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace iskhakov_d_linear_topology
