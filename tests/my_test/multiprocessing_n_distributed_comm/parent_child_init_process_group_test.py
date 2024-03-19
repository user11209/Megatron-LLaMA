import torch
import torch.multiprocessing as mp
import time

_GLOBAL_VARIABLE_ = 1

def main_worker(rank):
  c2p_pipe_p, c2p_pipe_c = mp.Pipe()
  subprocess = mp.Process(target=sub_worker, args=(1, c2p_pipe_c))
  subprocess.start()

  all_reduce_group = torch.distributed.init_process_group(
    backend="nccl", init_method="tcp://localhost:23456", world_size=2, rank=0
  )

  all_reduce_tensor = torch.tensor([1., 2.], device=torch.device("cuda:0"))
  another_reduce_tensor = torch.tensor([1., 2.], device=torch.device("cuda:1"))

  torch.distributed.all_reduce(all_reduce_tensor, group=all_reduce_group)
  print(all_reduce_tensor)
  all_reduce_tensor_c = c2p_pipe_p.recv()
  print(all_reduce_tensor_c)

  torch.distributed.new_group([0])
  print("successfully newed a group.")

  check_global_variable(0)
  global _GLOBAL_VARIABLE_
  _GLOBAL_VARIABLE_ = 3
  check_global_variable(0)
  return

def sub_worker(rank, c2p_pipe):
  all_reduce_group = torch.distributed.init_process_group(
    backend="nccl", init_method="tcp://localhost:23456", world_size=2, rank=1
  )
  
  all_reduce_tensor = torch.tensor([3., 0.5], device=torch.device("cuda:1"))
  torch.distributed.all_reduce(all_reduce_tensor, group=all_reduce_group)
  c2p_pipe.send(all_reduce_tensor)
  time.sleep(15)
  torch.distributed.new_group([0])

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