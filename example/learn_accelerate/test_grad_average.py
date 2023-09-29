"""
Whether gradient is averaged across process.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from accelerate import Accelerator

SINGLE = False
def main():
    class Net(nn.Module):
        """
        Make the gradient equal to input.
        Consider the batch_size, the backward grad is equal to input / micro_bs
        """
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(1, 4, bias = False)
            self.lin.weight.data.fill_(1.0)
        
        def forward(self, x):
            """
            Args:
                x: shape (bs, 1)
            """
            y = self.lin(x)
            loss = y.sum(dim = 1).mean()
            return loss
    
    model = Net()
    data = [torch.tensor([i], dtype = torch.float) for i in range(15)]
    
    optimizer = SGD(model.parameters(), lr = 1)
    if SINGLE:
        dataloader = DataLoader(data, batch_size = 4)
        for x in dataloader:
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            print(f'input: {x}, grad: {model.lin.weight.grad}')
            optimizer.step()
    else:
        accelerator = Accelerator()
        dataloader = DataLoader(data, batch_size = 2) 
        # assume num_proc=2, here dev_bs = tot_bs / num_proc
        model, dataloader, optimizer = accelerator.prepare(
            model, dataloader, optimizer
        )
        for x in dataloader:
            optimizer.zero_grad()
            loss = model(x)
            # loss.backward()
            accelerator.backward(loss)
            unwrap_model = accelerator.unwrap_model(model)
            print((f'rank: {accelerator.process_index},'
                   f'loss: {loss.detach().cpu().numpy()}'
                f' input: {x.detach().cpu().numpy()}, '
                f'grad: {unwrap_model.lin.weight.grad.detach().cpu().numpy()}'
            ))
            optimizer.step()


if __name__ == '__main__':
    main()