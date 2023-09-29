The `prepare.py` has the function `get_model_optimizer_dataloader` to return:
- model: BertForSequenceClassification, bert-base-uncased
- optimizer: adamw optimizer with lr=1e-5
- train_dataloader: yelp_review_full dataset with first 1000 samples.

`dl.py` show how data is split into dataloaders of different processes.

## Launch
```
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 {script_name.py}

accelerate launch --config_file {yaml_file_path} {script_name.py}
```

## Some Changes
### get the model state dict
Run `test_get_stat.py` to show the model parameters
```
accelerate launch --config_file config/ds.yaml test_get_stat.py
```
For deepspeed, the wrapped model and original model parameters is all empty. 
Should use `accelerate.get_state_dict(model)` to get the state dict (return None for not main process).

For DDP, it does matter which obj to call the function `state_dict`
