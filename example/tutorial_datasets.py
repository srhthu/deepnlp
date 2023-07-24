# %%
from datasets import (
    load_dataset_builder,
    load_dataset,
    list_metrics,
    load_metric
)
from transformers import AutoTokenizer
# %%
# Dataset information
ds_builder = load_dataset_builder("glue", "sst2")
print(ds_builder.info.description)
print(ds_builder.info.features)
print(ds_builder.info.splits)
# %%
# Load dataset
dataset = load_dataset("glue", "sst2", split="train")
all_dataset = load_dataset("glue", "sst2")

print(dataset[0])
# %%
# Preprocess
# Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer(dataset[0]["sentence"]))

def tokenization(example):
    return tokenizer(example["sentence"])

dataset = dataset.map(tokenization, batched=True)

# %%
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
dataset.format['type']
# %%
metrics_list = list_metrics()
print(len(metrics_list))
print(metrics_list[:10])
# %%
metric = load_metric('glue', 'mrpc')
references = [0, 1]
predictions = [0, 1]
print(metric.compute(predictions = predictions, references = references))
# %%
