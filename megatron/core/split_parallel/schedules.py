# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model, get_model_type

from .activation_agent import ActivationAgent, set_activationagent_warmup, get_activation_agent
from . import p2p_communication as split_p2p_communication
from megatron.core.tensor_parallel.logging import do_log

import code
import time
import os

# Types
Shape = Union[List[int], torch.Size]

def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device = out.device,
        dtype = out.dtype,
    )

def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = (output,),
        grad_tensors = (grad_output,),
        keep_graph = True,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )





def forward_step(forward_step_func,
                 data_iterator,
                 model,
                 num_microbatches,
                 input_tensor,
                 forward_data_store,
                 timers,
                 collect_non_loss_data=False,
                 autocast_dtype=torch.float,
                 enable_autocast=False):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if timers is not None:
        timers('forward-compute', log_level=2).start()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        output_tensor, loss_func = forward_step_func(data_iterator, model)

    if parallel_state.is_pipeline_last_stage() and parallel_state.get_split_model_parallel_rank() == 0:
        if not collect_non_loss_data:
            if parallel_state.get_split_model_parallel_world_size() == 1:
                output_tensor = loss_func(output_tensor)
                loss, loss_reduced = output_tensor
                output_tensor = loss / num_microbatches
                forward_data_store.append(loss_reduced)
            else:
                loss, loss_reduced = loss_func(output_tensor)
                loss_averaged = loss / num_microbatches
                forward_data_store.append(loss_reduced)
                output_tensor_grad = torch.autograd.grad(loss_averaged, output_tensor)
                # if using split model parallel, forward-gpu only does backward for loss. 
                # output_tensor_grad get returned and sent to backward-gpu.
                output_tensor = output_tensor_grad
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if timers is not None:
        timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)

    if parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(grad_scaler, input_tensor, output_tensor,
                  output_tensor_grad, model_type, timers, deallocate_pipeline_outputs=False):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if timers is not None:
        timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and grad_scaler is not None:
        output_tensor = grad_scaler(output_tensor[0])
    
    if deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        try:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0], retain_graph=True)
        except Exception as e:
            print(repr(e))
            print("================= printing shapes =====================")
            print(output_tensor[0].shape)
            print(output_tensor_grad[0].shape)
            assert 0

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
            parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if timers is not None:
        timers('backward-compute').stop()

    return input_tensor_grad

# caller see pipeline/schedules.py
def forward_backward_split(*,
                            forward_step_func,
                            data_iterator: Union[Iterator, List[Iterator]],
                            model: Union[torch.nn.Module, List[torch.nn.Module]],
                            num_microbatches: int,
                            dtype: torch.dtype,
                            tensor_shape: Optional[Shape] = None, # unused
                            decoder_seq_length: Optional[int] = None, # unused
                            grad_scaler: Callable = None,
                            sequence_parallel: bool = False, # unused
                            forward_only: bool = False,
                            timers: Callable = None,
                            collect_non_loss_data: bool = False,
                            enable_autocast: bool = False,
                            deallocate_pipeline_outputs: bool = False,
                            no_sync_func: Optional[Callable] = None,
                            grad_sync_func: Optional[Callable] = None, # unused
                            param_sync_func: Optional[Callable] = None, # unused
                            optimizer=None):
    get_activation_agent().ping_pong_check(tag="forward_backward_split")
    #! preparation before forwarding and backwarding
    #TODO by zhang: check whether all parts can be directly used
    if optimizer is not None:
        optimizer.gather_parameters(skip_if_not_stepped=True)

    if isinstance(model, list):
        assert len(model) == 1, \
            "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    # Disable async grad reductions
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Compute number of warmup microbatches.
    # num_warmup_microbatches = \
    #     (parallel_state.get_pipeline_model_parallel_world_size() -
    #      parallel_state.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = 1
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None

    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(rank=pipeline_rank-1,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    send_tensor_shapes = get_tensor_shapes(rank=pipeline_rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    split_rank = parallel_state.get_split_model_parallel_rank()
    if parallel_state.get_split_model_parallel_world_size() != 1:
        seq_length, micro_batch_size, hidden_size = tensor_shape
        split_grad_shape = [(micro_batch_size, hidden_size)]

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    #* forwarding and backwarding
    set_activationagent_warmup(True)
    output_tensor = forward_step(forward_step_func, data_iterator,
                                    model, num_microbatches, input_tensor, forward_data_store,
                                    timers, collect_non_loss_data, dtype, enable_autocast)
    set_activationagent_warmup(False)
    print("[rank ", split_rank, "]: finish warmup.")

    # if parallel_state.get_split_model_parallel_rank() == 0:
    #     with open("/Megatron-LLaMA/examples_of_zhang/log/log_0.txt", "w") as log_file:
    #         log_file.write(str(get_activation_agent().id2obj_dict))
    # else:
    #     with open("/Megatron-LLaMA/examples_of_zhang/log/log_1.txt", "w") as log_file:
    #         log_file.write(str(get_activation_agent().id2obj_dict))

    if parallel_state.get_split_model_parallel_rank() == 0:
        output_tensor_grad = forward_step(forward_step_func, data_iterator,
                                         model, num_microbatches, input_tensor, forward_data_store,
                                         timers, collect_non_loss_data, dtype, enable_autocast)
        print("[rank ", split_rank, "]: finish forward step.")
        if not forward_only:
            # wrong send_tensor_shape, check it again
            send_split(output_tensor_grad[0], split_grad_shape, timers=timers)
    else:
        if not forward_only:
            # wrong recv_tensor_shape, check it again
            output_tensor_grad = recv_split(split_grad_shape, dtype, timers=timers)
            print("================== checking what has been recved ===================")
            print(len(output_tensor_grad))
            print(output_tensor_grad[0].shape)
            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)

    print("[rank ", split_rank, "]: one iteration forward/backward succeeded!")
    time.sleep(20)
    #! by zhang: cleanup after forwarding and backwarding
    #TODO by zhang: check whether all parts can be directly used
    if optimizer is not None:
        optimizer.record_grad_accumulation_boundary()
    # Launch any remaining grad reductions
    if no_sync_context is not None:
        enable_grad_sync()
        if grad_sync_func is not None:
            grad_sync_func(model.parameters())

    return forward_data_store





def forward_step_preprocess(forward_step_func,
                 data_iterator,
                 model,
                 timers,
                 forward_stage_io,
                 autocast_dtype=torch.float,
                 enable_autocast=False):
    """
    preprocess of the model. including dataloader and VocabParallelEmbedding.
    """
    if timers is not None:
        timers('forward-compute', log_level=2).start()

    parallel_state.set_vocab_selective_split_parallel(True)

    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    assert len(forward_stage_io) == 0, "forward_stage_io is not empty at preprocess stage! {}".format(forward_stage_io)
    #@note somehow parallel_state.is_pipeline_last_stage() also need this.
    # for split_parallel_rank == 0 and pipeline_parallel_rank == last and pipeline_world_size > 1, \
    # input for postprocess will be None. Best way is to force preprocess here.
    if parallel_state.is_pipeline_first_stage():
        with context_manager:
            output_tensor = forward_step_func.preprocess(data_iterator, model, do_vocab_embedding=True)
            output_tensor_detach = output_tensor.detach()
            output_tensor_detach.requires_grad = True

        forward_stage_io["preprocess"] = [None, output_tensor]
    else:
        with context_manager:
            output_tensor_detach = forward_step_func.preprocess(data_iterator, model, do_vocab_embedding=False)

    parallel_state.set_vocab_selective_split_parallel(False)
    return output_tensor_detach

def forward_step_main(forward_step_func,
                 model,
                 input_tensor,
                 timers,
                 forward_stage_io,
                 autocast_dtype=torch.float,
                 enable_autocast=False):
    """
    main process of the model, without preprocessing and postprocessing.
    Note, input_tensor is the return value of forward_step_pre_process or is passed from the former pipeline stage.
    """
    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    if len(forward_stage_io) != 1:
        assert len(forward_stage_io) == 0, \
            "len(forward_stage_io) == {}, which is not allowed at forward_step_main."
        assert input_tensor != None, \
            "forward_step_main need to get input tensor somewhere, "\
            "currently we have empty forward_stage_io and empty input_tensor."
    set_input_tensor(input_tensor)

    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        output_tensor = forward_step_func.main_process(model)
        output_tensor_detach = output_tensor.detach()
        output_tensor_detach.requires_grad = True

    forward_stage_io["main"] = [input_tensor, output_tensor]
    return output_tensor_detach

def forward_step_postprocess(forward_step_func,
                              model,
                              input_tensor,
                              forward_stage_io,
                              forward_data_store,
                              num_microbatches,
                              timers,
                              collect_non_loss_data=False,
                              autocast_dtype=torch.float,
                              enable_autocast=False):
    """
    postprocess of the model. including lm_logit and loss.
    Note, input_tensor must be the output of forward_step_main.
    """
    output_tensor = None
    if parallel_state.is_pipeline_last_stage():
        assert input_tensor != None, \
            "at least provide something to tell what lm_output is like, need to build all-reduce buffer."

        if enable_autocast:
            context_manager = torch.autocast("cuda", dtype=autocast_dtype)
        else:
            context_manager = contextlib.nullcontext()
        
        parallel_state.set_vocab_selective_split_parallel(True)
        with context_manager:
            output_tensor, loss_func = forward_step_func.postprocess(model, input_tensor)
        parallel_state.set_vocab_selective_split_parallel(False)

        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

        # If T5 model (or other model with encoder and decoder)
        # and in decoder stack, then send encoder_hidden_state
        # downstream as well.
        model_type = get_model_type(model)

        forward_stage_io["postprocess"] = [input_tensor, output_tensor]

        assert not model_type == ModelType.encoder_and_decoder, \
            "please note, split model parallel does not support encoder_and_decoder like model yet."
        # please note, the return below is not necessary.
        if parallel_state.is_pipeline_stage_after_split() and \
                model_type == ModelType.encoder_and_decoder:
            return [output_tensor, input_tensor[-1]]
    
    if timers is not None:
        timers('forward-compute').stop()
    return output_tensor

def backward_step_one_stage(grad_scaler,
                            input_tensor,
                            output_tensor,
                            output_tensor_grad,
                            timers,
                            deallocate_pipeline_outputs=False):
    """backward process of the postprocess stage."""
    if timers is not None:
        timers('backward-compute', log_level=2).start()

    if input_tensor != None:
        input_tensor.retain_grad()
    if output_tensor_grad is None and grad_scaler is not None:
        output_tensor = grad_scaler(output_tensor)

    if deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    if input_tensor != None:
        input_tensor_grad = input_tensor.grad
    else:
        input_tensor_grad = None
    return input_tensor_grad

def backward_step_pre_process(grad_scaler, 
                              forward_stage_io, 
                              output_tensor_grad, 
                              model_type, 
                              timers, 
                              deallocate_pipeline_outputs=False):
    if timers is not None:
        timers('backward-compute', log_level=2).start()

    #@todo backward of preprocess is not necessarily called for some pipeline_parallel_rank.
    assert model_type != ModelType.encoder_and_decoder, \
        "split_model_parallel currently doesn't support encoder_and_decoder."
    parallel_state.set_vocab_selective_split_parallel(True)
    input_tensor = None
    output_tensor = forward_stage_io["preprocess"][1]
    input_tensor_grad = backward_step_one_stage(grad_scaler, input_tensor, output_tensor, 
                        output_tensor_grad, timers, deallocate_pipeline_outputs)
    parallel_state.set_vocab_selective_split_parallel(False)
    return None

def backward_step_main(grad_scaler, 
                       forward_stage_io, 
                       output_tensor_grad, 
                       model_type, 
                       timers, 
                       deallocate_pipeline_outputs=False):
    assert model_type != ModelType.encoder_and_decoder, \
        "split_model_parallel currently doesn't support encoder_and_decoder."
    input_tensor = forward_stage_io["main"][0]
    output_tensor = forward_stage_io["main"][1]
    input_tensor_grad = backward_step_one_stage(grad_scaler, input_tensor, output_tensor, 
                        output_tensor_grad, timers, deallocate_pipeline_outputs)
    return input_tensor_grad

def backward_step_post_process(grad_scaler, 
                               forward_stage_io, 
                               output_tensor_grad, 
                               model_type, 
                               timers, 
                               deallocate_pipeline_outputs=False):
    #@todo backward of postprocess is not necessarily called for some pipeline_parallel_rank.
    assert model_type != ModelType.encoder_and_decoder, \
        "split_model_parallel currently doesn't support encoder_and_decoder."
    parallel_state.set_vocab_selective_split_parallel(True)
    input_tensor = forward_stage_io["postprocess"][0]
    output_tensor = forward_stage_io["postprocess"][1]
    input_tensor_grad = backward_step_one_stage(grad_scaler, input_tensor, output_tensor, 
                        output_tensor_grad, timers, deallocate_pipeline_outputs)
    parallel_state.set_vocab_selective_split_parallel(False)

    if timers is not None:
        timers('backward-compute', log_level=2).stop()
    return input_tensor_grad

# caller see pipeline/scheduler.py
def forward_backward_split_selective(*,
                            forward_step_func,
                            data_iterator: Union[Iterator, List[Iterator]],
                            model: Union[torch.nn.Module, List[torch.nn.Module]],
                            num_microbatches: int,
                            dtype: torch.dtype,
                            tensor_shape: Optional[Shape] = None, # unused
                            decoder_seq_length: Optional[int] = None, # unused
                            grad_scaler: Callable = None,
                            sequence_parallel: bool = False, # unused
                            forward_only: bool = False,
                            timers: Callable = None,
                            collect_non_loss_data: bool = False,
                            enable_autocast: bool = False,
                            deallocate_pipeline_outputs: bool = False,
                            no_sync_func: Optional[Callable] = None,
                            grad_sync_func: Optional[Callable] = None, # unused
                            param_sync_func: Optional[Callable] = None, # unused
                            optimizer=None):
    assert parallel_state.get_split_model_parallel_world_size() > 1, \
        "You can only use selective_split_parallel with split world_size > 1!"
    get_activation_agent().ping_pong_check(tag="forward_backward_split")
    #! preparation before forwarding and backwarding
    #TODO by zhang: check whether all parts can be directly used
    if optimizer is not None:
        optimizer.gather_parameters(skip_if_not_stepped=True)

    if isinstance(model, list):
        assert len(model) == 1, \
            "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    # Disable async grad reductions
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Compute number of warmup microbatches.
    # num_warmup_microbatches = \
    #     (parallel_state.get_pipeline_model_parallel_world_size() -
    #      parallel_state.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = 1
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None

    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(rank=pipeline_rank-1,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    send_tensor_shapes = get_tensor_shapes(rank=pipeline_rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    split_rank = parallel_state.get_split_model_parallel_rank()
    if parallel_state.get_split_model_parallel_world_size() != 1:
        seq_length, micro_batch_size, hidden_size = tensor_shape
        split_grad_shape = [(micro_batch_size, hidden_size)]

    #TODO by zhang: forwarding and backwarding
    forward_stage_io = {}
    forward_stage_io_last = {}
    # do warmup-forward
    set_activationagent_warmup(True)
    preprocess_output = forward_step_preprocess(forward_step_func, data_iterator, model,
                                                timers, forward_stage_io, dtype, enable_autocast)
    if split_rank != 0:
        main_output = forward_step_main(forward_step_func, model, preprocess_output, timers,
                                        forward_stage_io, dtype, enable_autocast)
    else:
        main_output = torch.zeros_like(preprocess_output)
        main_output.requires_grad = True
    post_process_output = forward_step_postprocess(forward_step_func, model, main_output,
                                                    forward_stage_io, forward_data_store, num_microbatches,
                                                    timers, collect_non_loss_data, dtype, enable_autocast)
    set_activationagent_warmup(False)

    for k in num_microbatches_remaining:
        # symbol below: l-last, t-this, f-forward, b-backward, r-preprocess, m-main, o-postprocess
        # each iteration contains: lbo-tfr-[lbm & tfm]-lbr-tfo
        forward_stage_io_last = forward_stage_io
        forward_stage_io = {}
        # lbo
        main_output_grad = backward_step_post_process(grad_scaler, forward_stage_io_last, None, model_type,
                                                    timers, deallocate_pipeline_outputs)
        # tfr
        preprocess_output = forward_step_preprocess(forward_step_func, data_iterator, model,
                                                    timers, forward_stage_io, dtype, enable_autocast)
        if split_rank != 0:
            # lbm
            preprocess_output_grad = backward_step_main(grad_scaler, forward_stage_io_last, main_output_grad, model_type,
                                                        timers, deallocate_pipeline_outputs)
            main_output = torch.zeros_like(preprocess_output)
            main_output.requires_grad = True
        else:
            # tfm
            main_output = forward_step_main(forward_step_func, model, preprocess_output, timers,
                                            forward_stage_io, dtype, enable_autocast)
            preprocess_output_grad = torch.zeros_like(main_output_grad)
        # lbr
        backward_step_pre_process(grad_scaler, forward_stage_io_last, preprocess_output_grad, model_type,
                                timers, deallocate_pipeline_outputs)
        # tfo
        post_process_output = forward_step_postprocess(forward_step_func, model, main_output,
                                                        forward_stage_io, forward_data_store, num_microbatches,
                                                        timers, collect_non_loss_data, dtype, enable_autocast)
        print("[rank {}]: iteration {} forward/backward succeeded!".format(split_rank, k))

    # do cooldown-backward
    forward_stage_io_last = forward_stage_io
    forward_stage_io = {}
    main_output_grad = backward_step_post_process(grad_scaler, forward_stage_io_last, None, model_type,
                                                timers, deallocate_pipeline_outputs)
    if split_rank != 0:
        preprocess_output_grad = backward_step_main(grad_scaler, forward_stage_io_last, main_output_grad, model_type,
                                                    timers, deallocate_pipeline_outputs)
    else:
        preprocess_output_grad = torch.zeros_like(main_output_grad)
    backward_step_pre_process(grad_scaler, forward_stage_io_last, preprocess_output_grad, model_type,
                            timers, deallocate_pipeline_outputs)
    


    # if parallel_state.get_split_model_parallel_rank() == 0:
    #     with open("/Megatron-LLaMA/examples_of_zhang/log/log_0.txt", "w") as log_file:
    #         log_file.write(str(get_activation_agent().id2obj_dict))
    # else:
    #     with open("/Megatron-LLaMA/examples_of_zhang/log/log_1.txt", "w") as log_file:
    #         log_file.write(str(get_activation_agent().id2obj_dict))


    
    time.sleep(20)
    #! by zhang: cleanup after forwarding and backwarding
    #TODO by zhang: check whether all parts can be directly used
    if optimizer is not None:
        optimizer.record_grad_accumulation_boundary()
    # Launch any remaining grad reductions
    if no_sync_context is not None:
        enable_grad_sync()
        if grad_sync_func is not None:
            grad_sync_func(model.parameters())

    return forward_data_store

def get_tensor_shapes(*,
                      rank: int,
                      model_type: ModelType,
                      tensor_shape: Shape,
                      decoder_seq_length: int,
                      sequence_parallel: bool):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    assert (
        len(tensor_shape) == 3
    ), f"`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size] but {tensor_shape}"

    seq_length, micro_batch_size, hidden_size = tensor_shape

    if sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()

    if model_type == ModelType.encoder_and_decoder:
        if sequence_parallel:
            decoder_seq_length = decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()

        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    return tensor_shapes



def recv_forward(tensor_shapes, dtype, timers):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, dtype,
                                                                timers=timers))
    return input_tensors


def recv_backward(tensor_shapes, dtype, timers):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, dtype,
                                                                       timers=timers))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, timers=timers)


def send_backward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, timers=timers)


def send_forward_recv_backward(output_tensors, tensor_shapes, dtype, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, tensor_shape, dtype, timers=timers)
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, dtype, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
                input_tensor_grad, tensor_shape, dtype, timers=timers)
        input_tensors.append(input_tensor)
    return input_tensors

def send_split(send_tensors, tensor_shapes, timers):
    if not isinstance(send_tensors, list):
        send_tensors = [send_tensors]
    for (send_tensor, tensor_shape) in zip(send_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        split_p2p_communication.send_split(send_tensor, timers=timers)

def recv_split(tensor_shapes, dtype, timers):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(split_p2p_communication.recv_split(tensor_shape, dtype,
                                                                       timers=timers))
    return output_tensor_grads
