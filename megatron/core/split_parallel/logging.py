import os
import torch

_STORE_MID_TENSOR = False

def do_log(rank, tag, info=""):
  file_name = "/Megatron-LLaMA/examples_of_zhang/log/log_"+str(rank)+".txt"
  if os.path.exists(file_name):
    with open(file_name, "a") as log_file:
      log_file.write(tag+" "+str(info)+"\n")
  else:
    with open(file_name, "w") as log_file:
      log_file.write(tag+" "+str(info)+"!\n")

def store_mid_tensor(tag, tensor, rank=None):
  global _SOTRE_MID_TENSOR
  if _STORE_MID_TENSOR != True:
    return

  if rank == None:
    rank = torch.distributed.get_rank()
  dir_name = "/Megatron-LLaMA/examples_of_zhang/log/log_"+str(rank)
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)
  if not os.path.exists(os.path.join(dir_name, "mid_tensor")):
    os.mkdir(os.path.join(dir_name, "mid_tensor"))

  replicate_file_name_count = 0
  file_name = tag + str(replicate_file_name_count) + ".pt"
  while os.path.exists(os.path.join(dir_name, "mid_tensor", file_name)):
    replicate_file_name_count += 1
    file_name = tag + str(replicate_file_name_count) + ".pt"
  path = os.path.join(dir_name, "mid_tensor", file_name)
  torch.save(tensor, path)
  # print("[rank ", rank, "]: saving to path ", path)
  return path