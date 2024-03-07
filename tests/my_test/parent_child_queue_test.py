import torch
import torch.multiprocessing as mp

import time

def main_worker(rank):
  pipe_p2c_p, pipe_p2c_c = mp.Pipe()
  queue = mp.Manager().Queue()

  subprocess = mp.Process(target=sub_worker, args=(1, 5, pipe_p2c_c, queue))
  subprocess.start()

  for i in range(3):
    print(queue.get())

  subprocess.join()

def sub_worker(rank, init_value, pipe, queue):
  a = init_value
  for i in range(3):
    a += 1
    time.sleep(3*i+3)
    queue.put(a)
  return

if __name__ == '__main__':
  main_worker(0)