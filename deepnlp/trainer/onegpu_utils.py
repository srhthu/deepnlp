from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter


OPTIMIZER: Dict[str, torch.optim.Optimizer] = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}

def simple_data_collator(features: List) -> Dict[str, Any]:
    first = features[0]
    batch = {}
    for k,v in first.items():
        if not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


def infer_batch_sizes(
    batch_size:int, 
    device_batch_size:Optional[int] = None,
    use_cuda: Optional[bool] = None
):
    """
    Infer parameters related to batch_size.

    batch_size = device_batch_size * n_device * acc_step

    Return:
        device_batch_size
        n_device
        acc_step
    """
    # determin n_device
    if use_cuda is None:
        use_cuda = torch.cuda.is_available() # use if available
    n_device = torch.cuda.device_count() if use_cuda else 1

    if (device_batch_size is None or
        device_batch_size * n_device > batch_size):
        # do not need accumulation.
        device_batch_size, mod = divmod(batch_size, n_device)
        acc_step = 1
        iter_batch_size = device_batch_size * n_device
    else :
        # in the future, can decrease device_batch_size.
        iter_batch_size = device_batch_size * n_device
        acc_step, mod = divmod(batch_size, iter_batch_size)
    
    if mod != 0:
        raise ValueError((
            f'batch_size={batch_size} cannot be deviced exactly.'
            f' We get device={n_device}, acc_step={acc_step}, '
            f'device_batch_size={device_batch_size} and mod is {mod}')
        )
    
    assert batch_size == device_batch_size * n_device * acc_step

    return device_batch_size, n_device, acc_step


def default_feed(model: nn.Module, batch_data: Dict[str, torch.Tensor]):
    """
    return
        Dict:
            loss: torch.Tensor
            [other_keys]: object to print
    """
    outputs =  model(**batch_data)
    return outputs

def to_cuda(data):
    """
    Move input tensors to cuda.

    Args:
        data (dict or list)
    """
    def _cuda(x):
        return x.cuda() if isinstance(x, torch.Tensor) else x

    if isinstance(data, dict):
        cuda_data = {k:_cuda(v) for k,v in data.items()}
        
    elif isinstance(data, list):
        cuda_data = [_cuda(k) for k in data]
    else:
        raise ValueError(f'data should be an instance of dict or list, but get {type(data)}')
    return cuda_data

def to_number(x: torch.Tensor):
    """
    Return single element torch.Tensor as float else None
    """
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        return x.item() if x.reshape(-1).shape[0] == 1 else None
    else:
        return None

# customize for your project
def outputs_to_number(data: Dict):
    """
    Convert model outputs to python numbers.
    """
    num_data = {}
    for k,v in data.items():
        if isinstance(v, torch.Tensor):
            numb = to_number(v)
            if numb is not None:
                num_data[k] = numb
        elif isinstance(v, (int, float)):
            num_data[k] = v
    return num_data

def obj_to_str(obj, **kwargs):
    if isinstance(obj, (int, float, np.float64, np.int32, np.int64)):
        return numerical_to_str(obj)
    elif isinstance(obj, (tuple, list)):
        return '[' + ', '.join(map(obj_to_str, obj)) + ']'
    elif isinstance(obj, dict):
        sep = kwargs.get('sep', ' ')
        return sep.join([
            f'{k}: {obj_to_str(v, sep = sep)}' for k,v in obj.items()
        ])
    else:
        return str(obj)
        
def numerical_to_str(x: Union[int, float]):
    if abs(x) > 0.001 and abs(x) < 1000:
        return f'{x:.5g}'
    elif abs(x) < 1e-6:
        return '0.0'
    else:
        return f'{x:.3e}'


# metric recorder
class NumberAverager:
    """
    Maintain a list of numbers and calculate their average.
    """
    def __init__(self):
        self.values = []
    def append(self, x):
        self.values.append(x)
    def _reset(self):
        self.values = []
    def average(self, reset = True):
        ave = np.mean(self.values)
        if reset:
            self._reset()
        return ave

class DictNumberAverager:
    """
    A dict of cls NumberAverager
    """
    def __init__(self):
        self.records = {}
    
    def record(self, msg: dict):
        for name, value in msg.items():
            if not isinstance(value, (int, float)):
                # skip
                continue
            if name not in self.records:
                self.records[name] = NumberAverager()
            self.records[name].append(value)
    
    def average(self, reset = True):
        ave_dict = {name: k.average(reset) for name, k in self.records.items()}
        return ave_dict

