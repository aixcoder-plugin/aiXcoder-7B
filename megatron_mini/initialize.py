# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""

import random
import os
import time
import sys
import numpy as np
import torch
from datetime import timedelta

from megatron_mini import get_args
from megatron_mini.core import mpu, tensor_parallel
from megatron_mini.arguments import (parse_args, validate_args)
from megatron_mini.global_vars import set_global_variables


def initialize_megatron(extra_args_provider=None, args_defaults={},
                        ignore_unknown_args=False, allow_no_cuda=False, aix_config=None):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only 
    data processing. In general this arg should not be set unless you know 
    what you are doing.
    Returns a function to finalize distributed env initialization
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), 'Megatron requires CUDA.'

    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)

    validate_args(args, args_defaults, aix_config=aix_config)
        
    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()
        

    args = get_args()
    # Megatron's MPU is the master. Complete initialization right away.
    finish_mpu_init()
    _set_random_seed(14)
    # No continuation function
    return None


def _initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
    
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
    timeout = timedelta(minutes=args.dist_timeout)
    print(
        f"  > (rank={args.rank}) initializing process group: "
        f"world_size={args.world_size} "
        f"backend={args.distributed_backend} "
        f"init_method={init_method} ",
        f"init_timeout={timeout} ",
        flush=True, file=sys.stderr
    )

    # Call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        init_method=init_method,
        world_size=args.world_size, rank=args.rank,
        timeout=timeout)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            mpu.initialize_model_parallel(args.tensor_model_parallel_size,
                                           1,
                                           None,
                                           0)
            if args.rank == 0:
                print(f'> initialized tensor model parallel with size '
                      f'{mpu.get_tensor_model_parallel_world_size()}')
                print(f'> initialized pipeline model parallel with size '
                      f'{mpu.get_pipeline_model_parallel_world_size()}')


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed_))
