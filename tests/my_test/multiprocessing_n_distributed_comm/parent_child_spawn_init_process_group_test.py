import torch
import torch.multiprocessing as mp
import time

_GLOBAL_VARIABLE_ = 1

def main_worker(rank):
  c2p_pipe_p_0, c2p_pipe_c_0 = mp.Pipe()
  c2p_pipe_p_1, c2p_pipe_c_1 = mp.Pipe()
  subprocess = mp.spawn(sub_worker, args=(1, [c2p_pipe_c_0, c2p_pipe_c_1]), nprocs=2, join=False)

  world_group = torch.distributed.init_process_group(
    backend="nccl", init_method="tcp://localhost:23456", world_size=3, rank=0
  )
  all_reduce_group = torch.distributed.new_group([0,1])

  all_reduce_tensor = torch.tensor([1., 2.], device=torch.device("cuda:0"))

  print("checking GroupMember.WORLD: ", torch.distributed.distributed_c10d._get_default_group())
  torch.distributed.all_reduce(all_reduce_tensor, group=all_reduce_group)
  print(all_reduce_tensor)
  all_reduce_tensor_c_0 = c2p_pipe_p_0.recv()
  print(all_reduce_tensor_c_0)
  all_reduce_tensor_c_1 = c2p_pipe_p_1.recv()
  print(all_reduce_tensor_c_1)

  del all_reduce_tensor_c_0
  del all_reduce_tensor_c_1

  print("successfully newed a group.")

  check_global_variable(0)
  global _GLOBAL_VARIABLE_
  _GLOBAL_VARIABLE_ = 3
  check_global_variable(0)

  subprocess.join()
  return

def sub_worker(subprocess_id, rank, c2p_pipe):
  if subprocess_id == 0:
    world_group = torch.distributed.init_process_group(
      backend="nccl", init_method="tcp://localhost:23456", world_size=3, rank=1
    )
  else:
    world_group = torch.distributed.init_process_group(
      backend="nccl", init_method="tcp://localhost:23456", world_size=3, rank=2
    )
  print("world: ", torch.distributed.group.WORLD)
  all_reduce_group = torch.distributed.new_group([0,1])
  print("all_reduce: ", all_reduce_group)
  torch.distributed.group.WORLD = all_reduce_group
  print("world changing to: ", torch.distributed.group.WORLD)

  all_reduce_tensor = torch.tensor([3., 0.5], device=torch.device("cuda:1"))
  if subprocess_id == 0:
    torch.distributed.distributed_c10d._update_default_pg(all_reduce_group)
    torch.distributed.all_reduce(all_reduce_tensor)
    c2p_pipe[0].send(all_reduce_tensor)
    print("subprocess 0 send succeeded!")
  else:
    c2p_pipe[1].send(all_reduce_tensor)
    print("subprocess 1 send succeeded!")
  time.sleep(15)

  check_global_variable(1)
  global _GLOBAL_VARIABLE_
  _GLOBAL_VARIABLE_ = 2
  check_global_variable(1)

  return

def check_global_variable(rank):
  global _GLOBAL_VARIABLE_
  print(_GLOBAL_VARIABLE_, " from rank ", rank)

if __name__ == "__main__":
  main_worker(0)