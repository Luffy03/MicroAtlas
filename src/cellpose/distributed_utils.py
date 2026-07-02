"""
Distributed training utility tools
Provides initialization, cleanup, and state checking functions needed for DDP (DistributedDataParallel) training
"""
import os
import torch
import torch.distributed as dist


def setup_distributed(rank, world_size, master_addr='localhost', master_port='29500'):
    """
    Initialize distributed process group
    
    Args:
        rank: current process rank (0 to world_size-1)
        world_size: total number of processes (GPU count)
        master_addr: master node address
        master_port: master node port
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize process group, using NCCL backend (GPU communication)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set the GPU used by the current process
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    """Check if in a distributed environment"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get current process rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get total GPU count"""
    if is_distributed():
        return dist.get_world_size()
    return 1
