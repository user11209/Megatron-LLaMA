from functools import reduce
import operator
from typing import Optional, List, Union, Callable, Tuple

import torch

from megatron import core, get_args
from megatron.core.parallel_state import (
    get_split_model_parallel_group,
    get_split_model_parallel_rank,
    get_split_model_parallel_world_size,
    get_global_memory_buffer,
    initialize_model_parallel
)

import time
import code

import torch.multiprocessing as mp
from datetime import timedelta

# Types
Shape = Union[List[int], torch.Size]
_ACTIVATION_AGENT = None
_WARMUP_ACTIVATION_AGENT = False

#* function for the subprocess
def subprocess_schedule_buffer_transfer(some_zero, args, p2c_pipe, c2p_queue, done_event):
  # initialize nccl group, together with the main processes. this is because only one nccl group at a time is allowed.
  activation_comm_group = None
  torch.distributed.init_process_group(
    backend=args.distributed_backend,
    world_size=2*args.world_size, rank=args.rank + args.world_size,
    timeout=timedelta(minutes=args.distributed_timeout_minutes))
  initialize_model_parallel(args.tensor_model_parallel_size,
                              args.pipeline_model_parallel_size,
                              args.split_model_parallel_size,
                              args.virtual_pipeline_model_parallel_size,
                              args.pipeline_model_parallel_split_rank, is_parent_process=False)
  activation_comm_group = get_split_model_parallel_group()

  while True:
    get_ping = p2c_pipe.recv()
    if get_ping == 1:
      p2c_pipe.send(2)
    else:
      p2c_pipe.send(3)
      break

  id2obj_dict = p2c_pipe.recv()

  with open("/Megatron-LLaMA/examples_of_zhang/log/"+str(args.rank + args.world_size)+".txt", "w") as log_file:
    log_file.write("at least print something!\n")
    try:
      log_file.write("starting to get rank: ")
      rank = torch.distributed.get_rank()
      log_file.write(str(rank)+"\n")
      if rank == 2:
        verify_tensor = torch.tensor([rank], dtype=torch.float32, device=torch.device("cuda:0"))
      else:
        verify_tensor = torch.tensor([rank], dtype=torch.float32, device=torch.device("cuda:1"))
      log_file.write(str(verify_tensor)+"\n")
      torch.distributed.all_reduce(verify_tensor, group=activation_comm_group)
      log_file.write(str(verify_tensor)+"\n")
      verify_tensor_clone = verify_tensor.cpu()
      c2p_queue.put(verify_tensor_clone)
      log_file.write("queue put succeeded!\n")
    except Exception as e:
      log_file.write("got a failure: "+repr(e))

  write_file_name = "/Megatron-LLaMA/examples_of_zhang/log/log_" + str(rank) + ".txt"
  with open(write_file_name, "w") as log_file:
    log_file.write(str(id2obj_dict))

  done_event.wait()
  return

#TODO: each agent should belong to a layer. when a layer produces a tensor to be send, it is passed to ActivationAgent with a buffer name, it will be sent to the partner agent. the agent owns a subprocess(on init), which share tensor buffer with its parentprocess. One problem is, share memory tensors should be created before subprocesses are created, so they need to be created before transformer layers are called. That means it's necessary to fetch the tensor buffer before calling the layer and write the calculated values to the buffer directly.
class ActivationAgent:
  def __init__(self):

    self.id2obj_dict = {}

    # start subprocess on parent process.
    mp.set_start_method("spawn")
    self.p2c_pipe_p, p2c_pipe_c = mp.Pipe()
    self.c2p_queue = mp.Manager().Queue()
    self.done_event = mp.Event()
    args = get_args()

    # start the subprocess before torch.distributed.init_process_group
    self.schedule_subprocess = mp.spawn(subprocess_schedule_buffer_transfer, \
                                          args=(args, p2c_pipe_c, self.c2p_queue, self.done_event), join=False)

  def identify_activation_agent_role(self):
    self.is_sender_agent = (get_split_model_parallel_rank() == 0)
    #TODO: need to be changed if not only two ranks
    self.partner_rank = 0 if not self.is_sender_agent else 1


  def add_tensor_buffer_like(self, tensor_name, tensor_prototype, copy_count=8):
    self.ping_pong_check(tag="add_tensor_buffer_like")
    if not isinstance(tensor_prototype, dict):
      tensor_prototype_dict = {None: tensor_prototype}
    else:
      tensor_prototype_dict = tensor_prototype

    tensor_id_separate = {}
    for label, proto in tensor_prototype_dict.items():
      buffer_shape = proto.shape
      buffer_dtype = proto.dtype
      tensor_id_separate[label] = self.add_tensor_buffer(tensor_name, buffer_shape, buffer_dtype, copy_count)

    tensor_id = self.assign_id(tensor_id_separate)
    return tensor_id

  def add_tensor_buffer(self, tensor_name, buffer_shape, buffer_dtype, copy_count):
    '''
    call on layer init. preallocate data_buffer from GlobalMemoryBuffer. copy_count can now be manually set.
    the same tensor_name may be repeatedly used by different layers, so a tag is returned to help the layers to know who got which tensor slots, and it somehow seems tensor_name itself is NOT NEEDED at all. a method is needed to distinguish slots with the tag. a method is needed to ensure the uniqueness of the tag.
    '''
    assert _WARMUP_ACTIVATION_AGENT == True
    # allocate tensor
    tensor_buffer_list = []
    # valid range: occupied_range[0] to occupied_range[1] is occupied by some tensors. 
    # if occupied_range[0] > occupied_range[1], it means occupied_range[0] to copy_count and 0 to occupied_range[1].
    # note, if labeled like this, the position `occupied_range[0] - 1 % copy_count` must be empty.
    occupied_range = torch.zeros(2, dtype=torch.int32)
    valid_to_schedule = torch.zeros(copy_count, dtype=torch.bool)
    #? should the timer be placed on cpu or on gpu?
    timer = torch.zeros(copy_count, dtype=torch.float32)
    occupied_range.share_memory_()
    valid_to_schedule.share_memory_()
    timer.share_memory_()
    buffer_item = {"buffer": tensor_buffer_list, "timer": timer, 
                    "occupied_range": occupied_range, "valid_to_schedule": valid_to_schedule}
    # assign_id
    buffer_id = self.assign_id(buffer_item)

    # notice, the third arg of get_global_memory_buffer().get_tensor(...) is name, 
    # and the name need to be different for each different buffer. thus, we place it here to utilize buffer_id.
    for i in range(copy_count):
      tensor_buffer = get_global_memory_buffer().get_tensor(list(buffer_shape), buffer_dtype, (buffer_id, i))
      tensor_buffer.share_memory_()
      tensor_buffer_list.append(tensor_buffer)
    return buffer_id

  def set_tensor(self, tensor_id, tensor_value):
    self.dispatch_operation(tensor_id, self.set_tensor_inner, tensor_value)

  def get_empty_tensor(self, tensor_id, tensor_shape):
    return self.dispatch_operation(tensor_id, self.get_empty_tensor_inner, tensor_shape)

  def remote_get_tensor(self, tensor_id):
    return self.dispatch_operation(tensor_id, self.remote_get_tensor_inner)

  def takeover_tensor(self, tensor_id):
    self.dispatch_operation(tensor_id, self.takeover_tensor_inner)

  def abandon_tensor(self, tensor_id):
    self.dispatch_operation(tensor_id, self.abandon_tensor_inner)

  def dispatch_operation(self, tensor_id, inner_function, *args):
    buffer_item = self.id2obj_dict[tensor_id]
    if None in buffer_item.keys():
      # the target is a single tensor, buffer_item is like {None: tensor_id_inner}
      tensor_id_inner = buffer_item[None]
      ret = inner_function(tensor_id_inner, *args)
      return ret
    else:
      # the target is a dict, buffer_item is like {label_1: tensor_id_inner, label_2: tensor_id_inner}
      ret = {}
      for label in buffer_item:
        tensor_id_inner = buffer_item[label]
        args_inner = []
        for arg in args:
          args_inner.append(arg[label])
        ret_inner = inner_function(tensor_id_inner, *tuple(args_inner))
        ret[label] = ret_inner
      return ret

  def set_tensor_inner(self, tensor_id, tensor_value):
    '''
    call on layer forward of forward-GPU. set the tensor_value to tensor_id.
    '''
    buffer_item = self.id2obj_dict[tensor_id]
    empty_buffer_slot_index = self.get_empty_buffer_slot(buffer_item)
    buffer_item["buffer"][empty_buffer_slot_index].copy_(tensor_value)
    buffer_item["timer"][empty_buffer_slot_index] = time.time()
    buffer_item["occupied_range"][1] = (buffer_item["occupied_range"][1] + 1) % buffer_item["timer"].shape[0]
    buffer_item["valid_to_schedule"][empty_buffer_slot_index] = True
    return empty_buffer_slot_index

  def get_empty_tensor_inner(self, tensor_id, tensor_shape):
    '''
    call on layer forward of forward-GPU. return an empty tensor, which can be used to fill by the forwarding process.
    '''
    buffer_item = self.id2obj_dict[tensor_id]
    empty_buffer_slot_index = self.get_empty_buffer_slot(buffer_item)
    copy_count = buffer_item["timer"].shape[0]
    buffer_item["occupied_range"][1] = (buffer_item["occupied_range"][1] + 1) % copy_count
    return buffer_item["buffer"][empty_buffer_slot_index]

  def remote_get_tensor_inner(self, tensor_id):
    '''
    call on layer forward of backward-GPU at recomputation time.
    the remote tensor is by default the tensor at the index of `occupied_range[0]`.
    '''
    buffer_item = self.id2obj_dict[tensor_id]
    get_buffer_slot_index = buffer_item["occupied_range"][0]
    while True:
      if buffer_item["occupied_range"][0] == buffer_item["occupied_range"][1]:
        # the buffer is empty
        self.wait_for_schedule()
        continue
      else:
        break
    return buffer_item["buffer"][get_buffer_slot_index]

  def takeover_tensor_inner(self, tensor_id):
    """
    call on layer forward of forward-GPU. this must happen to a tensor from `get_empty_tensor`.
    the taken over tensor is by default the tensor at the index of `(occupied_range[1]-1)%copy_count`.
    """
    buffer_item = self.id2obj_dict[tensor_id]
    copy_count = buffer_item["timer"].shape[0]
    takenover_buffer_slot_index = (buffer_item["occupied_range"][1] - 1) % copy_count
    buffer_item["timer"][takenover_buffer_slot_index] = time.time()
    buffer_item["valid_to_schedule"][takenover_buffer_slot_index] = True

  def abandon_tensor_inner(self, tensor_id):
    """
    call on layer backward of backward-GPU. this must happen to a tensor from `remote_get_tensor`.
    the abandoned tensor is by default the tensor at the index of `occupied_range[0]`.
    """
    buffer_item = self.id2obj_dict[tensor_id]
    get_buffer_slot_index = buffer_item["occupied_range"][0]
    copy_count = buffer_item["timer"].shape[0]
    buffer_item["occupied_range"][0] = (buffer_item["occupied_range"][0] + 1) % copy_count
    return True

  def trigger_schedule(self):
    """
    remove all labels, namely timer, occupied_range and valid_to_schedule. 
    create a subprocess and run schedule_buffer_transfer.
    """
    for tensor_id, buffer_item in self.id2obj_dict.items():
      if "buffer" in buffer_item:
        buffer_item["occupied_range"].zero_()
        buffer_item["timer"].zero_()
        buffer_item["valid_to_schedule"].zero_()
    self.schedule_buffer_transfer()

  def ping_pong_check(self, info=1, tag=""):
    # info == 1 for temporary check, else for the last check.
    self.p2c_pipe_p.send(info)
    get_pong = self.p2c_pipe_p.recv()
    if info == 1:
      assert get_pong == 2
    else:
      assert get_pong == 3
    print("ping-pong check succeeded at " + tag + "!")
    return

  def schedule_buffer_transfer(self):
    """
    schedule the transfer of all buffers. A subprocess will be opened to do the 
    transfer without blocking the computation of the main process.
    """
    self.ping_pong_check(info=3, tag="the last stage")
    self.p2c_pipe_p.send(self.id2obj_dict)
    print("id2obj_dict send succeeded!")
    verify_tensor = self.c2p_queue.get()
    print("checking verify tensor from rank ", get_split_model_parallel_rank(), " : ", verify_tensor)
    del verify_tensor
    self.done_event.set()
    self.schedule_subprocess.join()
    assert 0

  def wait_for_schedule(self):
    """
    wait for schedule! the main process wait for the subprocess to free an arbitrary buffer slot. 
    Wake up at the request from the subprocess.
    """
    pass
    
  def assign_id(self, obj):
    tensor_id = len(self.id2obj_dict)
    self.id2obj_dict[tensor_id] = obj
    return tensor_id

  def get_empty_buffer_slot(self, buffer_item):
    copy_count = buffer_item["timer"].shape[0]
    while True:
      occupied_range = buffer_item["occupied_range"]
      if (occupied_range[0]-occupied_range[1]) % copy_count == 1:
        # the buffer is full
        self.wait_for_schedule()
        continue
      else:
        break
    return occupied_range[1] % copy_count
    


def init_activation_agent():
  #TODO: confirm its is_sender_agent and partner_rank, where to do this
  global _ACTIVATION_AGENT
  _ACTIVATION_AGENT = ActivationAgent()
  pass

def get_activation_agent():
  assert _ACTIVATION_AGENT != None
  return _ACTIVATION_AGENT

def identify_activation_agent_role():
  assert _ACTIVATION_AGENT != None
  _ACTIVATION_AGENT.identify_activation_agent_role()

def set_activationagent_warmup(set_warmup):
  global _WARMUP_ACTIVATION_AGENT
  global _ACTIVATION_AGENT
  _WARMUP_ACTIVATION_AGENT = set_warmup
  if set_warmup == False:
    _ACTIVATION_AGENT.trigger_schedule()

def is_activationagent_warmup():
  return _WARMUP_ACTIVATION_AGENT
