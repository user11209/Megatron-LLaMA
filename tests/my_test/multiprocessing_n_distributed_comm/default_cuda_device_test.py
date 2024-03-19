import torch
import torch.multiprocessing as mp

import time

def worker(subprocess_id):
  print("from child: ", torch.cuda.current_device())
  return

def parent():
  torch.cuda.set_device(torch.device("cuda:1"))
  print("from parent: ", torch.cuda.current_device())

  time.sleep(5)
  mp.spawn(worker, args=(), join=True)

if __name__ == "__main__":  
  parent()