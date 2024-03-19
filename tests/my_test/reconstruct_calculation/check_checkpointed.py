import torch
import os

_CHECK_ID = 284
_BASE_RANK = 1

def check_id(index=_CHECK_ID):
  path_base = "/Megatron-LLaMA/examples_of_zhang/log/log_"
  file_name = str(index)+".pt"
  paths = []
  for rank in range(4):
    paths.append( os.path.join(path_base+str(rank), file_name) )

  values = []
  for rank in range(4):
    values.append( torch.load(paths[rank]) )

  print("Checking id ", _CHECK_ID)

  flag = True
  device = values[_BASE_RANK].device
  for rank in range(4):
    diff = values[rank].float().to(device) - values[_BASE_RANK].float()
    checksum = torch.sum( diff**2 )
    if checksum == 0:
      print("rank ", rank, " value is the same as rank ", str(_BASE_RANK))
    else:
      flag = False
      max_abs = torch.max(torch.abs(diff))
      print("rank ", rank, " failed, error is ", checksum, ", and max_abs is ", max_abs, " in amount of ", torch.max(torch.abs(values[_BASE_RANK].float())))

def check_pt_file(file_0, file_1):
  print("checking ", file_0, file_1)
  tensor_0 = torch.load(file_0)
  tensor_1 = torch.load(file_1)
  print("    mse: ", torch.sum((tensor_0 - tensor_1.to(tensor_0.device))**2))
  print("    max: ", torch.max(torch.abs(tensor_0 - tensor_1.to(tensor_0.device))))
  print("    avg: ", torch.max(torch.abs(tensor_0)))

check_id()
# check_pt_file("/Megatron-LLaMA/examples_of_zhang/log/log_0/mid_tensor/core_attn_output2.pt", "/Megatron-LLaMA/examples_of_zhang/log/log_3/mid_tensor/core_attn_output0.pt")
# check_pt_file("/Megatron-LLaMA/examples_of_zhang/log/log_0/mid_tensor/row_split_matmul_output2.pt", "/Megatron-LLaMA/examples_of_zhang/log/log_3/mid_tensor/row_split_matmul_output0.pt")
# check_pt_file("/Megatron-LLaMA/examples_of_zhang/log/log_0/mid_tensor/row_split_reduced_output0.pt", "/Megatron-LLaMA/examples_of_zhang/log/log_1/mid_tensor/row_split_reduced_output0.pt")