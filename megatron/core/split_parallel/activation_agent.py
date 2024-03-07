from functools import reduce
import operator
from typing import Optional, List, Union, Callable, Tuple

import torch

from megatron import core, get_args
from megatron.core.parallel_state import (
    get_split_model_parallel_group,
    get_split_model_parallel_group_cpu,
    get_split_model_parallel_rank,
    get_split_model_parallel_world_size,
    get_global_memory_buffer,
    initialize_model_parallel
)

import time
import code

import torch.multiprocessing as mp
from datetime import timedelta

from .logging import do_log

# Types
Shape = Union[List[int], torch.Size]
_ACTIVATION_AGENT = None
_WARMUP_ACTIVATION_AGENT = False

#* function for the subprocess
def subprocess_schedule_buffer_transfer(some_zero, args, p2c_pipe, p_c_queue, c_p_queue, current_device):
  parallel_rank = 500
  try:
    torch.cuda.set_device(current_device)
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
    activation_comm_group_cpu = get_split_model_parallel_group_cpu()

    while True:
      get_ping = p2c_pipe.recv()
      if get_ping == 1:
        p2c_pipe.send(2)
      else:
        p2c_pipe.send(3)
        break

    id2obj_dict = p2c_pipe.recv()
    torch.distributed.barrier(group=activation_comm_group)

    parallel_rank = torch.distributed.get_rank()
    split_parallel_rank = torch.distributed.get_rank(activation_comm_group)

    if split_parallel_rank == 0:
      while True:
        buffer_id = p_c_queue.get()
        do_log(parallel_rank, "the first to be send is", buffer_id)
        queue_size = p_c_queue.qsize()
        # tell its partner how many members are to be sent. #? not always be 3 when split_model_parallel_size>2
        torch.distributed.isend(torch.tensor([queue_size], dtype=torch.int32, device=torch.cuda.current_device()), \
                                3, group=activation_comm_group)
        do_log(parallel_rank, "seems the current number of elements to send is", queue_size+1)
        buffer_id_list = [buffer_id]
        device_list = []

        send_tensor_slot_index = id2obj_dict[buffer_id]["occupied_range"][0]
        #? when get_empty_tensor is allowed later, the assertion need to be changed into some wait on p_c_queue
        assert id2obj_dict[buffer_id]["valid_to_schedule"][send_tensor_slot_index] == True, \
          "Seems this slot is not valid to schedule yet, that is buffer_id "+str(buffer_id)+" and slot index "+str(send_tensor_slot_index)
        send_tensor = id2obj_dict[buffer_id]["buffer"][send_tensor_slot_index]
        #? the third arg should not always be 3 when split_model_parallel_size>2
        send_ops = []
        send_ops_cpu = []
        if send_tensor.device == torch.device("cpu"):
          send_ops_cpu.append(torch.distributed.P2POp(torch.distributed.isend, send_tensor, 3, 
                                          group=activation_comm_group_cpu))
          device_list.append("cpu")
        else:
          send_ops.append(torch.distributed.P2POp(torch.distributed.isend, send_tensor, 3, 
                                          group=activation_comm_group))
          device_list.append("cuda")
        for i in range(queue_size):
          buffer_id = p_c_queue.get()

          buffer_id_list.append(buffer_id)
          # find all available tensors in buffer_id, send them.
          send_tensor_slot_index = id2obj_dict[buffer_id]["occupied_range"][0]
          #? when get_empty_tensor is allowed later, the assertion need to be changed into some wait on p_c_queue
          assert id2obj_dict[buffer_id]["valid_to_schedule"][send_tensor_slot_index] == True, \
            "Seems this slot is not valid to schedule yet, that is buffer_id "+str(buffer_id)+" and slot index "+str(send_tensor_slot_index)
          send_tensor = id2obj_dict[buffer_id]["buffer"][send_tensor_slot_index]
          #? the third arg should not always be 3 for split_model_parallel_size>2
          if send_tensor.device == torch.device("cpu"):
            send_ops_cpu.append(torch.distributed.P2POp(torch.distributed.isend, send_tensor, 3, 
                                          group=activation_comm_group_cpu))
            device_list.append("cpu")
          else:
            send_ops.append(torch.distributed.P2POp(torch.distributed.isend, send_tensor, 3, 
                                          group=activation_comm_group))
            device_list.append("cuda")

        do_log(parallel_rank, "trying to send", buffer_id_list)

        reqs_cpu = torch.distributed.batch_isend_irecv(send_ops_cpu)
        reqs = torch.distributed.batch_isend_irecv(send_ops)

        req_num = 0
        req_num_cpu = 0
        for buffer_id, device_type in zip(buffer_id_list, device_list):
          if device_type == "cpu":
            reqs_cpu[req_num_cpu].wait()
            req_num_cpu += 1
          else:
            reqs[req_num].wait()
            req_num += 1
          copy_count = id2obj_dict[buffer_id]["timer"].shape[0]
          id2obj_dict[buffer_id]["occupied_range"][0] = (id2obj_dict[buffer_id]["occupied_range"][0]+1)%copy_count
          c_p_queue.put(buffer_id)    

          do_log(parallel_rank, "successfully send", buffer_id)
    else:
      next_recv = 0
      while True:
        # tell its partner how many members are to be sent. #? not always be 3 when split_model_parallel_size>2
        queue_size_tensor = torch.tensor([0], dtype=torch.int32, device=torch.cuda.current_device())
        size_req = torch.distributed.irecv(queue_size_tensor, 2, group=activation_comm_group)
        size_req.wait()
        queue_size = queue_size_tensor[0].item() + 1
        do_log(parallel_rank, "seems the current number of element to recv is ", queue_size)
        # find all tensors in buffer_id, recv them in order.
        recv_ops = []
        recv_ops_cpu = []
        buffer_id_list = []
        device_list = []
        #TODO: find a way to get a number of queue_size buffer_id's
        while len(recv_ops) + len(recv_ops_cpu) < queue_size:
          buffer_id = next_recv
          do_log(parallel_rank, "checking buffer id ", buffer_id)
          if "buffer" in id2obj_dict[buffer_id]:
            do_log(parallel_rank, "    going to constructing recv op for id", buffer_id)
            buffer_id_list.append(buffer_id)
            next_recv = (buffer_id+2) if (buffer_id+2)<len(id2obj_dict) else 0
            copy_count = id2obj_dict[buffer_id]["timer"].shape[0]
            #? somehow it's best to wait for the parent process to clarify that this buffer still has empty slot
            while True:
              occupied_range = id2obj_dict[buffer_id]["occupied_range"]
              if (occupied_range[0] - occupied_range[1])%copy_count == 1:
                # the buffer is full, don't recv yet
                do_log(parallel_rank, "    full buffer: ", buffer_id)
                p_c_queue.get()
              else:
                break
            empty_buffer_slot_index = id2obj_dict[buffer_id]["occupied_range"][1]
            buffer_tensor = id2obj_dict[buffer_id]["buffer"][empty_buffer_slot_index]
            if buffer_tensor.device == torch.device("cpu"):
              do_log(parallel_rank, "    constructing recv op cpu for id", buffer_id)
              device_list.append("cpu")
              recv_ops_cpu.append(torch.distributed.P2POp(torch.distributed.irecv, buffer_tensor, 2, activation_comm_group_cpu))
            else:
              do_log(parallel_rank, "    constructing recv op for id", buffer_id)
              device_list.append("cuda")
              recv_ops.append(torch.distributed.P2POp(torch.distributed.irecv, buffer_tensor, 2, activation_comm_group))
          
        
        do_log(parallel_rank, "trying to recv", buffer_id_list)
        reqs_cpu = torch.distributed.batch_isend_irecv(recv_ops_cpu)
        do_log(parallel_rank, "    waiting on cpu ...")
        reqs = torch.distributed.batch_isend_irecv(recv_ops)
        do_log(parallel_rank, "    waiting on cuda")

        req_num = 0
        req_num_cpu = 0
        for (buffer_id, device_type) in zip(buffer_id_list, device_list):
          if device_type == "cpu":
            do_log(parallel_rank, "trying to recv on cpu ", buffer_id)
            reqs_cpu[req_num_cpu].wait()
            req_num_cpu += 1
            do_log(parallel_rank, "    successfully recved on cpu ", buffer_id)
          else:
            do_log(parallel_rank, "trying to recv on cuda ", buffer_id)
            reqs[req_num].wait()
            req_num += 1
            do_log(parallel_rank, "    successfully recved on cuda ", buffer_id)
          id2obj_dict[buffer_id]["occupied_range"][1] = (id2obj_dict[buffer_id]["occupied_range"][1]+1)%copy_count
          c_p_queue.put(buffer_id)

          do_log(parallel_rank, "    successfully wrote ", buffer_id)
  except Exception as e:
    do_log(parallel_rank, "something went wrong: ", repr(e))
  return

#TODO: each agent should belong to a layer. when a layer produces a tensor to be send, it is passed to ActivationAgent with a buffer name, it will be sent to the partner agent. the agent owns a subprocess(on init), which share tensor buffer with its parentprocess. One problem is, share memory tensors should be created before subprocesses are created, so they need to be created before transformer layers are called. That means it's necessary to fetch the tensor buffer before calling the layer and write the calculated values to the buffer directly.
class ActivationAgent:
  def __init__(self):

    self.id2obj_dict = {}
    self.assign_id_offset = 0

    # start subprocess on parent process.
    mp.set_start_method("spawn")
    self.p2c_pipe_p, p2c_pipe_c = mp.Pipe()
    self.p_c_queue = mp.Manager().Queue()
    self.c_p_queue = mp.Manager().Queue()
    args = get_args()

    # start the subprocess before torch.distributed.init_process_group
    self.schedule_subprocess = mp.spawn(subprocess_schedule_buffer_transfer, \
                                          args=(args, p2c_pipe_c, self.p_c_queue, self.c_p_queue, torch.cuda.current_device()), join=False)

  def identify_activation_agent_role(self):
    self.is_sender_agent = (get_split_model_parallel_rank() == 0)
    #TODO: need to be changed if not only two ranks
    self.partner_rank = 0 if not self.is_sender_agent else 1


  def add_tensor_buffer_like(self, tensor_name, tensor_prototype, copy_count=8):
    if not isinstance(tensor_prototype, dict):
      tensor_prototype_dict = {None: tensor_prototype}
    else:
      tensor_prototype_dict = tensor_prototype

    tensor_id_separate = {}
    for label, proto in tensor_prototype_dict.items():
      buffer_shape = proto.shape
      buffer_dtype = proto.dtype
      buffer_device = None if proto.device!=torch.device("cpu") else torch.device("cpu")
      tensor_id_separate[label] = self.add_tensor_buffer(tensor_name, buffer_shape, buffer_dtype, copy_count, buffer_device=buffer_device)

    tensor_id = self.assign_id(tensor_id_separate)
    return tensor_id

  def add_tensor_buffer(self, tensor_name, buffer_shape, buffer_dtype, copy_count, buffer_device=None):
    '''
    call on layer init. preallocate data_buffer from GlobalMemoryBuffer. copy_count can now be manually set.
    the same tensor_name may be repeatedly used by different layers, so a tag is returned to help the layers to know who got which tensor slots, and it somehow seems tensor_name itself is NOT NEEDED at all. a method is needed to distinguish slots with the tag. a method is needed to ensure the uniqueness of the tag.
    '''
    if _WARMUP_ACTIVATION_AGENT == False:
      return self.assign_id(None)

    # allocate tensor
    tensor_buffer_list = []
    # occupied range: occupied_range[0] to occupied_range[1] is occupied by some tensors. 
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
      tensor_buffer = get_global_memory_buffer().get_tensor(list(buffer_shape), buffer_dtype, (buffer_id, i), device=buffer_device)
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
    buffer_item["valid_to_schedule"][empty_buffer_slot_index] = True
    # notice, "occupied_range" need to be changed last, or it will change the value of empty_buffer_slot_index
    buffer_item["occupied_range"][1] = (buffer_item["occupied_range"][1] + 1) % buffer_item["timer"].shape[0]
    global _WARMUP_ACTIVATION_AGENT
    if not _WARMUP_ACTIVATION_AGENT:
      # print("\tchecking: ", tensor_id, empty_buffer_slot_index, " is going to be set!")
      self.p_c_queue.put(tensor_id)
    return empty_buffer_slot_index

  def get_empty_tensor_inner(self, tensor_id, tensor_shape):
    '''
    call on layer forward of forward-GPU. return an empty tensor, which can be used to fill by the forwarding process.
    '''
    buffer_item = self.id2obj_dict[tensor_id]
    empty_buffer_slot_index = self.get_empty_buffer_slot(buffer_item)
    copy_count = buffer_item["timer"].shape[0]
    ret_tensor = buffer_item["buffer"][empty_buffer_slot_index]
    buffer_item["occupied_range"][1] = (buffer_item["occupied_range"][1] + 1) % copy_count
    return ret_tensor

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
    self.p_c_queue.put(tensor_id)
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
    print("id2obj_dict send to activation agent succeeded!")

  def wait_for_schedule(self):
    """
    wait for schedule! the main process wait for the subprocess to free an arbitrary buffer slot. 
    Wake up at the request from the subprocess.
    """
    message = self.c_p_queue.get()
    
  def assign_id(self, obj):
    max_tensor_id = len(self.id2obj_dict)
    if _WARMUP_ACTIVATION_AGENT == False:
      tensor_id = self.assign_id_offset
      self.assign_id_offset = (self.assign_id_offset + 1) % max_tensor_id
      return tensor_id
    else:
      self.id2obj_dict[max_tensor_id] = obj
      return max_tensor_id

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
    return occupied_range[1]
    


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
