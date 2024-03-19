import torch
import torch.multiprocessing as mp

import time

def main_worker(rank):
  a = torch.tensor([1.], dtype=torch.float32).cuda()
  a.share_memory_()

  pipe_p2c_p, pipe_p2c_c = mp.Pipe()
  queue_c2p = mp.Manager().Queue()

  # subprocess = mp.Process(target=sub_worker, args=(1, None, pipe_p2c_c))
  # subprocess.start()
  subprocess = mp.spawn(sub_worker, args=(1, None, pipe_p2c_c, queue_c2p), nprocs=1, join=False)

  pipe_p2c_p.send(a)
  for i in range(12):
    print(a)
    time.sleep(5)

  print("parent process escaping!")
  value = queue_c2p.get()
  print("got ", value, " from child process!")
  subprocess.join()

def sub_worker(sub_process_id, rank, arg_tensor, pipe, queue):
  print("got sub_process_id ", sub_process_id)
  arg_tensor = pipe.recv()
  print("got arg_tensor on ", arg_tensor.device)
  while True:
    a = arg_tensor
    a[0] += 1
    time.sleep(10)
    if a[0] > 7:
      break

  queue.put("__message__")
  return

if __name__ == '__main__':
  main_worker(0)