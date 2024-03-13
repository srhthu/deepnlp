"""
Explore the communication of processes
"""
import torch
import torch.distributed as dist
from accelerate import Accelerator

def test_broadcast_obj(accelerator: Accelerator):
    # Note: Process group initialization omitted on each rank.
    if accelerator.process_index == 0:
        # Assumes world_size of 3.
        objects = ["foo", 12, {1: 2}] # any picklable object
    else:
        objects = [None, None, None]
    # Convert to cuda as backend is nccl
    device = torch.device(f"cuda:{accelerator.process_index}")
    dist.broadcast_object_list(objects, src=0, device=device)
    
    print(f'Process {accelerator.process_index}: {objects}')

def test_gather_obj(accelerator: Accelerator):
    local_rank = accelerator.process_index
    # data = {'pred': torch.tensor([local_rank]).cuda()}
    data = {'pred': torch.tensor([local_rank]).cuda(), 'id': '123'}
    output = [None for _ in range(accelerator.num_processes)] if accelerator.is_main_process else None
    dist.gather_object(data, output)
    print(f'Process {local_rank}, {output}')


def main():
    acc = Accelerator()
    
    # test_broadcast_obj(acc)

    test_gather_obj(acc)


if __name__ == '__main__':
    main()