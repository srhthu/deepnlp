import torch
from accelerate import Accelerator
from transformers import Trainer

from tqdm import tqdm
from prepare import get_model_optimizer_dataloader

def main():
    max_step = 1
    # get Bert model and a text classification dataset
    model, optimizer, training_dataloader = get_model_optimizer_dataloader()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: 1.0)
    accelerator = Accelerator()

    def log(txt):
        print(f'Local rank {accelerator.process_index}: {txt}')

    # wrap model and optimizers, dataloaders (bs=8, num_sample=1000)
    # refer to dl.py to see how dataloaders are prepared
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, scheduler
    )
    stat = accelerator.get_state_dict(model)
    if accelerator.process_index==0:
        torch.save(stat, './tmp/state_dict_start.bin')

    device = accelerator.device
    log(device)
    accelerator.wait_for_everyone()
    tot_loss = 0.0

    accelerator.main_process_first()

    epc_iterator = tqdm(training_dataloader, ncols = 80, disable = not accelerator.process_index)
    for step, batch in enumerate(epc_iterator):
        # compute loss
        outputs = model(**batch)
        loss = outputs.loss
        if accelerator.process_index == 0:
            epc_iterator.write(str(loss))
        # loss backward
        accelerator.backward(loss)
        # update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # print loss
        log(str(loss))
        # tot_loss = accelerator.gather_for_metrics(loss.detach())
        tot_loss = accelerator.gather(loss.detach())
        log('tot loss: ' + str(tot_loss))

        if step + 1 >= max_step:
            break

if __name__ == "__main__":
    main()