import torch
from torch.utils.data import DataLoader
import accelerate
from accelerate import Accelerator
import os


def main():
    # deepspeed_fields_from_accelerate_config = os.environ.get("ACCELERATE_CONFIG_DS_FIELDS", "").split(",")
    # print(deepspeed_fields_from_accelerate_config)
    accelerator = Accelerator()
    net = torch.nn.Linear(2, 2)
    # optimizer = torch.optim.AdamW(net.parameters(), lr = 1e-4)
    optimizer = accelerate.utils.DummyOptim
    model, _ = accelerator.prepare(net, optimizer)

    if accelerator.process_index==0:
        print('Original net:')
        print(net.state_dict())
        print('Wrapped model:')
        print(type(model))
        print(model.state_dict())
        unwrap_model = accelerator.unwrap_model(model)
        print('Unwrapped model:')
        print(type(unwrap_model))
        print(unwrap_model.state_dict())
        
        print('Use get_state_dict')
    
    state_dict = accelerator.get_state_dict(model)
    print(accelerator.process_index, state_dict)

if __name__ == '__main__':
    main()