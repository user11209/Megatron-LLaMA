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
  all_reduce_group_cpu = torch.distributed.new_group([1,2], backend="gloo")
  all_reduce_group = torch.distributed.new_group([1,2])

  subprocess.join()
  return

def sub_worker(subprocess_id, rank, c2p_pipe):
  if subprocess_id == 0:
    torch.cuda.set_device(torch.device("cuda:0"))
    world_group = torch.distributed.init_process_group(
      backend="nccl", init_method="tcp://localhost:23456", world_size=3, rank=1
    )
  else:
    torch.cuda.set_device(torch.device("cuda:1"))
    world_group = torch.distributed.init_process_group(
      backend="nccl", init_method="tcp://localhost:23456", world_size=3, rank=2
    )
  all_reduce_group_cpu = torch.distributed.new_group([1,2], backend="gloo")
  all_reduce_group = torch.distributed.new_group([1,2])

  if subprocess_id == 0:
    send_tensor_cpu = torch.tensor([1., 2.], device=torch.device("cpu"))
    send_tensor = torch.tensor([3., 0.5], device=torch.device("cuda:0"))

    send_ops_cpu = [torch.distributed.P2POp(torch.distributed.isend, send_tensor_cpu, 2, group=all_reduce_group_cpu)]
    send_ops = [torch.distributed.P2POp(torch.distributed.isend, send_tensor, 2, group=all_reduce_group)]

    reqs_cpu = torch.distributed.batch_isend_irecv(send_ops_cpu)
    for req in reqs_cpu:
      req.wait()
    reqs = torch.distributed.batch_isend_irecv(send_ops)
    for req in reqs:
      req.wait()

    print("subprocess 0 send succeeded!")
  else:
    recv_tensor_cpu = torch.tensor([0., 0.], device=torch.device("cpu"))
    recv_tensor = torch.tensor([0., 0.], device=torch.device("cuda:1"))

    recv_ops_cpu = [torch.distributed.P2POp(torch.distributed.irecv, recv_tensor_cpu, 1, group=all_reduce_group_cpu)]
    recv_ops = [torch.distributed.P2POp(torch.distributed.irecv, recv_tensor, 1, group=all_reduce_group)]
    
    reqs_cpu = torch.distributed.batch_isend_irecv(recv_ops_cpu)
    for req in reqs_cpu:
      req.wait()
    reqs = torch.distributed.batch_isend_irecv(recv_ops)
    for req in reqs:
      req.wait()

    print("subprocess 1 recv succeeded, got ", str(recv_tensor_cpu), str(recv_tensor))
    
  time.sleep(15)
  return

if __name__ == "__main__":
  main_worker(0)