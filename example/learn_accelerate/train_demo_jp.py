# %%
import os
from accelerate.utils import write_basic_config
# %%
write_basic_config()  # Write a config file
os._exit(00)  # Restart the notebook
# %%
import os, re, torch, PIL
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

from accelerate import Accelerator
from accelerate.utils import set_seed

from prepare import get_model_optimizer_dataloader