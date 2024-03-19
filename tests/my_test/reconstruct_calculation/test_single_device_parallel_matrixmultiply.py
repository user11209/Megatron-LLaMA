import sys
sys.path.append("/Megatron-LLaMA")

from megatron.core.tensor_parallel.random import CudaRNGStatesTracker
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.random import _CUDA_RNG_STATE_TRACKER
from megatron.core.tensor_parallel.random import checkpoint
from megatron.core import tensor_parallel

from megatron.data.gpt_dataset import build_train_valid_test_datasets

import torch

from tests.my_test.test_utilities import Utils
rank = Utils.rank

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

def test_chunked_transformer():
  Utils.initialize_model_parallel(tensor_model_parallel_size=3, pipeline_model_parallel_size=1)

  train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            train_valid_test_datasets_provider)

  encoder = ParallelTransformer(
                self.init_method,
                output_layer_init_method,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=False,
                post_process=False,
            )