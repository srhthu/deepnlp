"""
Show how dataloader work under distributed training.
"""
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

"""
total=50, num_proc = 4, bs = 3
tot_bs = 12
50 + 10(copied) = 12 * 5 = 60

[0,12), [12, 24)
Samples are first split into groups of total_batch_size
In each macro batch, samples are split into groups of num_processes
"""
def main():
    split_batches = True
    accelerator = Accelerator(split_batches=split_batches)
    NP=accelerator.num_processes
    DEV_BS=3
    if split_batches:
        BS=NP*DEV_BS
    else:
        BS=DEV_BS
    # BS is the total batch size
    TOTAL = 50

    data = list(range(TOTAL))
    dataloader = DataLoader(data, batch_size = BS)
    dataloader = accelerator.prepare(dataloader)
    
    host = list(iter(dataloader))
    lork = accelerator.process_index
    print(f'Rank {lork}\n{host}\n')
    for bi, batch in enumerate(host):
        idx = DEV_BS*NP*bi + DEV_BS*lork
        assert batch[0].item() in [idx, idx-TOTAL]

if __name__ == '__main__':
    main()