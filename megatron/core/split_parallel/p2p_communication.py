# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from functools import reduce
import operator
from typing import Optional, List, Union, Callable, Tuple

import torch

from megatron import core
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_next_rank,
    get_split_model_parallel_forward_rank,
    get_split_model_parallel_backward_ranks,
)

# Types
Shape = Union[List[int], torch.Size]

def _communicate_shapes(tensor_send_backward):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.
    """
    recv_forward_shape_tensor = None
    if tensor_send_backward is None:
        recv_forward_shape_tensor = torch.empty((3),
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
    else:
        send_backward_shape_tensor = torch.tensor(tensor_send_backward.size(),
                                              device=torch.cuda.current_device(),
                                              dtype=torch.int64)

    ops = []
    if tensor_send_backward is None:
        recv_forward_op = torch.distributed.P2POp(
            torch.distributed.irecv, recv_forward_shape_tensor,
            mpu.get_split_model_parallel_forward_rank())
        ops.append(recv_forward_op)
    else:
        #? much more complicated when using split_model_parallel_size > 2
        send_backward_op = torch.distributed.P2POp(
            torch.distributed.isend, send_backward_shape_tensor,
            mpu.get_split_model_parallel_backward_ranks()[0])
        ops.append(send_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # To protect against race condition when using batch_isend_irecv().
    # should take this out once the bug with batch_isend_irecv is resolved.
    torch.cuda.synchronize()

    recv_forward_shape = [0, 0, 0]
    if recv_forward_shape_tensor is not None:
        recv_forward_shape = recv_forward_shape_tensor.tolist()

    return recv_forward_shape


def _communicate(*, tensor_send_backward: Optional[torch.Tensor],
                 tensor_shape: Shape,
                 dtype: Optional[torch.dtype],
                 variable_seq_lengths: bool = False,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Arguments:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        dtype (torch.dtype, required if either recv_{prev,next} is True):
            this must be the type of the tensors that will be
            received, will typically be params_dtype, but in the case
            of fp32 residual connections might be torch.float.

        variable_seq_lengths (bool, optional, default=False):
            Support for variable sequence lengths across
            microbatches. Setting this communicates the size of
            tensors during pipeline parallelism communication, because
            of this extra overhead it should only be set if the
            sequence length is not constant during training.

        use_ring_exchange_p2p (bool, optional, default = False):
            Use custom ring_exchange kernel instead of
            torch.distributed.batch_isend_irecv(). Requires custom
            built torch with torch.distributed.ring_exchange.


    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_forward = None

    if not variable_seq_lengths:
        recv_forward_shape = tensor_shape
    else:
        recv_forward_shape = \
            _communicate_shapes(tensor_send_backward)

    if tensor_send_backward == None:
        if dtype is None:
            raise RuntimeError("dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_forward = torch.empty(recv_forward_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)

    # Send tensors to backward-device
    ops = []
    if tensor_send_backward is not None:
        send_backward_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_backward,
            get_split_model_parallel_backward_ranks()[0])
        ops.append(send_backward_op)
    else:
        recv_forward_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_forward,
            get_split_model_parallel_forward_rank())
        ops.append(recv_forward_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    # To protect against race condition when using batch_isend_irecv().
    # User should assert that we have a modern enough PyTorch to not need this
    torch.cuda.synchronize()

    return tensor_recv_forward


def recv_split(tensor_shape: Shape,
                  dtype: torch.dtype,
                  timers: Callable = None) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if core.parallel_state.get_split_model_parallel_world_size() != 1:
        if timers is not None:
            timers('backward-recv', log_level=2).start()
        output_tensor_grad = _communicate(
            tensor_send_backward=None,
            tensor_shape=tensor_shape,
            dtype=dtype)
        if timers is not None:
            timers('backward-recv').stop()
    return output_tensor_grad


def send_split(output_tensor: torch.Tensor,
                 timers: Callable = None) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if core.parallel_state.get_split_model_parallel_world_size() != 1:
        if timers is not None:
            timers('forward-send', log_level=2).start()
        _communicate(
            tensor_send_backward=output_tensor,
            tensor_shape=None,
            dtype=None)
        if timers is not None:
            timers('forward-send').stop()