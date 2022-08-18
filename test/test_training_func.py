import os
import sys
sys.path.append(os.getcwd())
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.models.bert import BertForSequenceClassification

from deepnlp.training_func import train_one_epoch, BatchSize, train_multiple_epochs


class DS(Dataset):
    def __init__(self, len = 200):
        self.len = len

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 1000, (512,)),
            'attention_mask': torch.ones((512,)),
            'labels': torch.randint(0,2, (1,)).item()
        }

def compute_loss(model, batch):
    outs = model(**batch)
    logits = outs['logits']
    preds = torch.argmax(logits, dim = -1)
    acc = (preds == batch['labels']).to(torch.float32).mean()
    return {'loss': outs['loss'], 'acc': acc, 'preds': preds, 'labels': batch['labels']}

def eval_metric(outputs):
    preds = outputs['preds']
    labels = outputs['labels']
    acc = (preds == labels).astype(np.float32).mean()
    return {'acc': acc}

def main():

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(10)
    hdl = logging.StreamHandler()
    
    hdl.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(hdl)
    
    dataset = DS()
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-5)

    bs = BatchSize(64, device_batch_size=16)
    print(bs)

    # train_one_epoch(
    #     model, bs, dataset, optim,
    #     compute_loss = compute_loss, logging_steps = 5, 
    #     logger = logger, show_bar = True
    # )
    args = {
        'batch_size': 16,
        'device_batch_size': 8,
        'eval_batch_size': 32,
        'output_dir': None,
        'early_stop': True,
        'early_stop_metric': 'acc',
        'num_epoch': 5,
        'optim_name': 'adamw',
        'lr': 0.01,
        'logging_steps': 5,
        'output_dir': './tmp'
    }
    epoch_eval_results = train_multiple_epochs(args, model, DS(100), [DS(100), DS(100)], compute_loss = compute_loss, eval_feed_fn = compute_loss, eval_metric_fn = eval_metric, logger = logger)

    print(epoch_eval_results)


if __name__ == '__main__':
    main()