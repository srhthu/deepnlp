# import context
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepnlp.trainer.acc_trainer import TrainingArgs, AccTrainer
from accelerate import PartialState


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5,5)
    
    def forward(self, x, y):
        logits = self.lin(x)
        loss = F.cross_entropy(logits, y, reduction = 'mean')
        time.sleep(0.2)
        return {'loss': loss, 'logits': logits}

# -----Config-----

# training stops on num_epoch
# num_samples = 100
# args = TrainingArgs(16, device_batch_size = 8, num_epoch= 3, logging_steps=2)
# dev_dataset = None

# training stops on max_steps
# num_samples = 100
# args = TrainingArgs(16, device_batch_size = 8, max_steps = 50, logging_steps=2)
# dev_dataset = None

# training + evaluation@step
# num_samples = 100
# args = TrainingArgs(16, device_batch_size = 8, max_steps = 50, logging_steps=2, eval_steps = 10)
# dev_dataset = [{
#     'x': np.random.standard_normal(5).astype(np.float32),
#     'y': np.random.randint(0, 5)
#     } for _ in range(50)]

# training + evaluation@epoch
# num_samples = 100
# args = TrainingArgs(16, device_batch_size = 8, max_steps = 50, logging_steps=2, eval_epochs = 2)
# dev_dataset = [{
#     'x': np.random.standard_normal(5).astype(np.float32),
#     'y': np.random.randint(0, 5)
#     } for _ in range(50)]

# training + multi evaluation
num_samples = 100
args = TrainingArgs(16, device_batch_size = 8, max_steps = 50, logging_steps=2, eval_steps = 10)
dev_dataset = [{
    'x': np.random.standard_normal(5).astype(np.float32),
    'y': np.random.randint(0, 5)
    } for _ in range(50)]

# For evaluate
class MyTrainer(AccTrainer):
    def compute_preds(self, model, batch):
        outs = model(**batch)
        return {'logits': outs['logits']}

def compute_metrics(outputs, inputs):
    logits = outputs['logits']
    preds = np.argmax(logits, axis = 1)
    targets = inputs['y']
    acc = (preds == targets).astype(np.float32).mean()
    return {'acc': acc}

# -----End Config-----

# generate samples
dataset = [{
    'x': np.random.standard_normal(5).astype(np.float32),
    'y': np.random.randint(0, 5)
    } for _ in range(num_samples)]
model = ToyModel()
trainer = MyTrainer(
    args, model, dataset, 
    eval_dataset = dev_dataset,
    compute_metrics=compute_metrics
)

print(f'Local rank: {PartialState().process_index}')
trainer.train()

if __name__ == '__main__':
    ...