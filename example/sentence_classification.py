# %%
import os
import sys
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
# %%
dataset = load_dataset('glue', 'mrpc', split = 'train')

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

dataset = dataset.map(encode, batched=True)
print(dataset[0])
# %%
dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
# %%
