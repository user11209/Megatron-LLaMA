# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib

import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from megatron.core.tensor_parallel import get_cuda_rng_tracker

from .utils import (
    split_tensor_into_1d_equal_chunks,
    gather_split_1d_tensor,
)

from megatron.core.utils import safely_set_viewless_tensor_data

from .activation_agent import ActivationAgent, get_activation_agent, is_activationagent_warmup

class SplitCheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
    """
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        assert not distribute_saved_activations, "this is not taken into consideration yet"
        ctx.run_function = run_function
        ctx.distribute_saved_activations \
            = distribute_saved_activations

        fwd_cpu_rng_state = torch.get_rng_state()
        fwd_cuda_rng_state = torch.cuda.get_rng_state()
        fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Copy the rng states.
        activation_agent = get_activation_agent()
        if is_activationagent_warmup():
            fwd_cpu_rng_state_num = activation_agent.add_tensor_buffer_like("fwd_cpu_rng_state", fwd_cpu_rng_state)
            fwd_cuda_rng_state_num = activation_agent.add_tensor_buffer_like("fwd_cuda_rng_state", fwd_cuda_rng_state)
            fwd_cuda_rng_state_tracker_num = activation_agent.add_tensor_buffer_like("fwd_cuda_rng_state_tracker", fwd_cuda_rng_state_tracker)
            ctx.num_dict = {"fwd_cpu_rng_state": fwd_cpu_rng_state_num, 
                            "fwd_cuda_rng_state": fwd_cuda_rng_state_num, 
                            "fwd_cuda_rng_state_tracker": fwd_cuda_rng_state_tracker_num}
        
        activation_agent.set_tensor(ctx.num_dict["fwd_cpu_rng_state"], fwd_cpu_rng_state)
        activation_agent.set_tensor(ctx.num_dict["fwd_cuda_rng_state"], fwd_cuda_rng_state)
        activation_agent.set_tensor(ctx.num_dict["fwd_cuda_rng_state_tracker"], fwd_cuda_rng_state_tracker)

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True))

        # Store everything.
        ctx.arg_list_without_tensor = list(map(lambda x: None if isinstance(x, torch.Tensor) else x, args))
        ctx.num_dict["args"] = []
        for arg_index, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                if is_activationagent_warmup():
                    arg_name = activation_agent.add_tensor_buffer_like(str(arg_index), arg)
                    ctx.num_dict["args"].append(arg_name)
                else:
                    arg_name = ctx.num_dict["args"][arg_index]
                activation_agent.set_tensor(arg_name, arg)
            else:
                if is_activationagent_warmup():
                    ctx.num_dict["args"].append(None)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        assert not ctx.distribute_saved_activations, "this is not taken into consideration yet"
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")
        activation_agent = get_activation_agent()
        arg_name_list = ctx.num_dict["args"]
        inputs = ctx.arg_list_without_tensor.copy()
        for arg_index, arg_name in enumerate(arg_name_list):
            if arg_name == None:
                # this arg is not a tensor. inputs get it from ctx.arg_list_without_tensor.
                continue
            # this arg is a tensor. inputs get it from somewhere else, namely activation_agent.
            arg_name = arg_name_list[arg_index]
            arg = activation_agent.remote_get_tensor(arg_name)
            inputs[arg_index] = arg
        inputs = tuple(inputs)

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        fwd_cpu_rng_state = activation_agent.remote_get_tensor(ctx.num_dict["fwd_cpu_rng_state"])
        fwd_cuda_rng_state = activation_agent.remote_get_tensor(ctx.num_dict["fwd_cuda_rng_state"])
        fwd_cuda_rng_state_tracker = activation_agent.remote_get_tensor(ctx.num_dict["fwd_cuda_rng_state_tracker"])
        torch.set_rng_state(fwd_cpu_rng_state)
        _set_cuda_rng_state(fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args, retain_graph=False)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)

        for arg_index, arg_name in enumerate(arg_name_list):
            if arg_name != None:
                activation_agent.abandon_tensor(arg_name)

        return (None, None) + grads

def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return SplitCheckpointFunction.apply(function,
                                    distribute_saved_activations, *args)
