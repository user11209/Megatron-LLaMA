from functools import reduce
import operator
from typing import Optional, List, Union, Callable, Tuple

import torch

from megatron import core
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_next_rank,
)

# Types
Shape = Union[List[int], torch.Size]
_ACTIVATION_AGENT = None
_WARMUP_ACTIVATION_AGENT = False

#TODO: each agent should belong to a layer. when a layer produces a tensor to be send, it is passed to ActivationAgent with a buffer name, it will be sent to the partner agent. the agent owns a subprocess(on init), which share tensor buffer with its parentprocess. One problem is, share memory tensors should be created before subprocesses are created, so they need to be created before transformer layers are called. That means it's necessary to fetch the tensor buffer before calling the layer and write the calculated values to the buffer directly.
class ActivationAgent:
  def __init__(self, is_sender_agent, partner_rank):
    self.is_sender_agent = is_sender_agent
    self.partner_rank = partner_rank

  def add_tensor_buffer(self, tensor_name, buffer_shape, buffer_dtype, copy_count):
    '''
    call on layer init. preallocate data_buffer from GlobalMemoryBuffer. copy_count can now be manually set.
    the same tensor_name may be repeatedly used by different layers, so a tag is returned to help the layers to know who got which tensor slots. a method is needed to distinguish slots with the tag. a method is needed to ensure the uniqueness of the tag.
    '''
    assert _WARMUP_ACTIVATION_AGENT == True
    pass

  def set_tensor(self, tensor_name, tensor_value):
    '''
    call on layer forward of forward-GPU. set the tensor_value to tensor_name.
    '''
    pass

  def get_empty_tensor(self, tensor_shape, tensor_name):
    '''
    call on layer forward of forward-GPU. return an empty tensor, which can be used to fill by the forwarding process.
    '''

  def remote_get_tensor(self, tensor_name):
    '''
    call on layer forward of backward-GPU at recomputation time.
    '''
    pass

  def takeover_tensor(self, tensor_name):
    pass

  def abandon_tensor(self, tensor_name):
    pass

  def schedule_buffer_transfer(self):
    pass

def init_activation_agent(is_sender_agent, partner_rank):
  #TODO: confirm its is_sender_agent and partner_rank, where to do this
  global _ACTIVATION_AGENT
  _ACTIVATION_AGENT = ActivationAgent(is_sender_agent, partner_rank)
  pass

def get_activation_agent():
  assert _ACTIVATION_AGENT != None
  return _ACTIVATION_AGENT

def set_activationagent_warmup(set_warmup):
  global _WARMUP_ACTIVATION_AGENT
  _WARMUP_ACTIVATION_AGENT = set_warmup

def is_activationagent_warmup():
  return _WARMUP_ACTIVATION_AGENT
