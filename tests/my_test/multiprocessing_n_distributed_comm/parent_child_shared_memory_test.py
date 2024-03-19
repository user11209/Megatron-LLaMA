import torch
import torch.multiprocessing as mp

import time

def main_worker(rank):
  a = torch.tensor([1.], dtype=torch.float32)
  a.share_memory_()

  pipe_p2c_p, pipe_p2c_c = mp.Pipe()

  subprocess = mp.Process(target=sub_worker, args=(1, a, pipe_p2c_c))

  subprocess.start()
  for i in range(20):
    print(a)
    time.sleep(5)

  subprocess.join()

def sub_worker(rank, arg_tensor, pipe):
  while True:
    a = arg_tensor
    a[0] += 1
    time.sleep(10)
    if a[0] > 7:
      break
  return

if __name__ == '__main__':
  main_worker(0)