# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension:")

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True, manual_pre_process=False, manual_post_process=False):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        manual_pre_process=manual_pre_process,
        manual_post_process=manual_post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}
    

class ForwardStep:
    def __init__(self):
        self.lm_output_buffer = None
        self.timers = None

    def __call__(self, data_iterator, model):
        """Forward step."""
        args = get_args()
        if self.timers == None:
            self.timers = get_timers()

        # Get the batch.
        self.timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
        self.timers('batch-generator').stop()

        output_tensor = model(tokens, position_ids, attention_mask,
                            labels=labels)

        return output_tensor, partial(loss_func, loss_mask)

    def preprocess(self, data_iterator, model, do_vocab_embedding=True):
        if self.timers == None:
            self.timers = get_timers()

        # Get the batch.
        self.timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
        self.timers('batch-generator').stop()

        self.tokens = tokens
        self.position_ids = position_ids
        self.labels = labels
        self.loss_mask = loss_mask
        self.attention_mask = attention_mask
        # this returned value need to be set to transformer model using model.set_input_tensor.
        if do_vocab_embedding:
            pre_process_func = get_attr_wrapped_model(model, "model_pre_process")
            lm_input = pre_process_func(tokens, position_ids)
        return lm_input

    def main_process(self, model):
        #@note self.tokens, self.position_ids, self.labels are just placeholders here. can be None if nothing goes wrong.
        lm_output = model(self.tokens, self.position_ids, self.attention_mask,
                        labels=self.labels)
        return lm_output

    def postprocess(self, model, lm_output, is_proto=False):
        if lm_output == None:
            assert self.lm_output_buffer != None, \
                "post_process need at least a proto to construct an all-reduce buffer."
            lm_output = self.lm_output_buffer
        elif is_proto and self.lm_output_buffer == None:
            self.lm_output_buffer = lm_output
        post_process_func = get_attr_wrapped_model(model, "model_post_process")
        output_tensor = post_process_func(lm_output, self.labels)
        return output_tensor, partial(loss_func, self.loss_mask)

forward_step = ForwardStep()

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
