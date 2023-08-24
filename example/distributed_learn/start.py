import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist

mp_vars = ['LOCAL_RANK', 'RANK', 'LOCAL_WORLD_SIZE', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
print({k:v for k,v in os.environ.items() if k in mp_vars})
print(sys.argv)

if os.environ['LOCAL_RANK'] == '0':
    # print(os.environ)
    ...

dist.init_process_group('nccl')
# gloo do not support multi nodes.

rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ['LOCAL_RANK'])
if rank == 0:
    print('GPUs: ', torch.cuda.device_count())

def demo_all_reduce():
    if local_rank == 0:
        print('Example of all_reduce')
    tensor = torch.Tensor([rank]).cuda(local_rank)
    print(f'Before all_reduce: Rank {rank} {tensor}')
    dist.all_reduce(tensor)
    print(f'After all_reduce: Rank {rank} {tensor}')
    dist.barrier()
    if local_rank == 0:
        print('='*30)

def demo_reduce():
    if local_rank == 0:
        print('Example of reduce')
        # time.sleep(5) # sleep do not cause errors
    tensor = torch.Tensor([rank]).cuda(local_rank)
    print(f'Before reduce: Rank {rank} {tensor}')
    dist.reduce(tensor, 0) # all workers should report the tensor to worker_0
    print(f'After reduce: Rank {rank} {tensor}')
    dist.barrier()
    if local_rank == 0:
        print('='*30)

def demo_all_gather():
    if local_rank == 0:
        print('Example of all_gather')
    tensor = torch.Tensor([rank]).cuda(local_rank)
    print(f'Before all_gather: Rank {rank} {tensor}')
    tensor_list = [torch.zeros(tensor.shape).cuda(local_rank) for _ in range(world_size)]
    # It is critical to make tensor_list on local_rank GPU for nccl
    dist.all_gather(tensor_list, tensor)
    print(f'After all_gather: Rank {rank} {tensor_list}')
    dist.barrier()
    if local_rank == 0:
        print('='*30)

def demo_gather():
    if local_rank == 0:
        print('Example of gather')
    tensor = torch.Tensor([rank]).cuda(local_rank)
    print(f'Before gather: Rank {rank} {tensor}')
    if rank == 0:
        tensor_list = [torch.zeros(tensor.shape).cuda(local_rank) for _ in range(world_size)]
    else:
        tensor_list = None # critical to set to None for other workers
    dist.gather(tensor, tensor_list, dst = 0)
    print(f'After gather: Rank {rank} {tensor_list}')
    dist.barrier()
    if rank == 0:
        ave = torch.mean(torch.cat(tensor_list))
        print(f'Average: {ave}')
    if local_rank == 0:
        print('='*30)

# demo_all_reduce()
demo_reduce()
# demo_all_gather()
demo_gather()
"""
torchrun --nproc_per_node=2  start.py

torchrun --nproc_per_node=2 --master_addr 172.28.176.19 --master_port 15432  start.py

torchrun --nproc_per_node=2 --master_addr 172.28.176.19 --master_port 15432 --nnodes 2 --node_rank 0 start.py
"""