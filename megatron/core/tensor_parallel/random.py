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
    get_split_model_parallel_world_size,
    get_split_model_parallel_rank,
)

from .utils import (
    split_tensor_into_1d_equal_chunks,
    gather_split_1d_tensor,
)

from megatron.core.utils import safely_set_viewless_tensor_data

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME_BASE = 'model-parallel-rng'
_MODEL_PARALLEL_RNG_TRACKER_NAME = None

def get_model_parallel_rng_tracker_name():
    global _MODEL_PARALLEL_RNG_TRACKER_NAME
    return _MODEL_PARALLEL_RNG_TRACKER_NAME

def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)



class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        if name == None:
            global _MODEL_PARALLEL_RNG_TRACKER_NAME
            name = _MODEL_PARALLEL_RNG_TRACKER_NAME
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-tensor-model-parallel regions.
        tensor-model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    global _MODEL_PARALLEL_RNG_TRACKER_NAME
    if get_split_model_parallel_world_size() == 1:
        # when split model parallel is not activated, fall back to the old method.
        _MODEL_PARALLEL_RNG_TRACKER_NAME = _MODEL_PARALLEL_RNG_TRACKER_NAME_BASE
        tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()
        _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME,
                                tensor_model_parallel_seed)
    else:
        split_world_size = get_split_model_parallel_world_size()
        split_rank = get_split_model_parallel_rank()
        print("by zhang: {} trying to set _MODEL_PARALLEL_RNG_TRACKER_NAME!".format(split_rank))
        if split_rank != 0:
            _MODEL_PARALLEL_RNG_TRACKER_NAME = _MODEL_PARALLEL_RNG_TRACKER_NAME_BASE + str(split_rank-1)
        else:
            _MODEL_PARALLEL_RNG_TRACKER_NAME = []
            for backward_index in range(split_world_size-1):
                _MODEL_PARALLEL_RNG_TRACKER_NAME.append(_MODEL_PARALLEL_RNG_TRACKER_NAME_BASE + str(backward_index))
            # NOTE: the forward GPU of split model parallel should not use default tracker name!
        for backward_index in range(split_world_size-1):
            tensor_model_parallel_seed = offset + backward_index
            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME_BASE + str(backward_index),
                                    tensor_model_parallel_seed)


def check_rng_state(tag = ""):
    if get_tensor_model_parallel_rank() != 0:
        return
    log_dir_name = "/Megatron-LLaMA/examples_of_zhang/log"
    import os
    if not os.path.exists(os.path.join(log_dir_name, "rng_log")):
        os.mkdir(os.path.join(log_dir_name, "rng_log"))
    with open(os.path.join(log_dir_name, "log_recorder.txt"), "r") as record_file:
        log_count = int(record_file.readline())
    with open(os.path.join(log_dir_name, "log_recorder.txt"), "w") as record_file:
        record_file.write(str(log_count + 1))
    
    if log_count >= 100:
        return
    with open(os.path.join(log_dir_name, "rng_log", str(log_count)+".txt"), "w") as rng_log_file:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()[_MODEL_PARALLEL_RNG_TRACKER_NAME]

        def hash_func(state):
            return torch.sum((255-state)*241)

        log_str = tag + "\n============ cpu state ===========\n" + str(hash_func(cpu_rng_state))
        log_str +=      "\n=========== cuda state ===========\n" + str(hash_func(cuda_rng_state))
        log_str +=      "\n========== tracker state =========\n" + str(hash_func(cuda_rng_state))
        rng_log_file.write(log_str)

class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
    """
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations \
            = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

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
        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

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
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    if get_split_model_parallel_world_size() > 1:
        import megatron.core.split_parallel as split_parallel
        return split_parallel.checkpoint(function, distribute_saved_activations, *args)
    else:
        return CheckpointFunction.apply(function,
                                    distribute_saved_activations, *args)
