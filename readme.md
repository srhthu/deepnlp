# Introduction
`Deepnlp` is a pytorch based deep learning library for NLP. It provides off-the-shelf commonly used classes/functions for training a deep learning model.

# Main Modules
## nn_modules
`nn_modules` implements common neural networks in NLP, e.g., cnn (Seq2SeqVec_CNN) and rnn (Seq2SeqVec_RNN). The `StackedModel` support config file based initialization so you can build an end-to-end model quickly. The following example build a cnn-lstm model for sentence classification.
```
from deepnlp.nn_modules import StackedModel
config = [
    [
        "embedding",
        [10000,100]
    ],
    [
        "seq2seqvec_cnn",
        {
            "input_size": 100,
            "hidden_size": 128
        }
    ],
    [
        "seq2seqvec_rnn",
        {
            "rnn_type": "lstm",
            "input_size": 128,
            "hidden_size": 128
        }
    ],
    [
        "seqpool",
        ["mean",]
    ]
    [
        "mlp",
        {
            "hidden_sizes": [256,128,115],
            "last_activation": false
        }
    ]
]
model = StackedModel(config)
print(model)
"""
ModuleList(
  (0): Embedding(10000, 100)
  (1): Seq2SeqVec_CNN(
    (cnn): Conv1d(100, 128, kernel_size=(3,), stride=(1,), padding=(1,))
    (act): Tanh()
  )
  (2): Seq2SeqVec_RNN(
    (rnn): LSTM(128, 128, batch_first=True, bidirectional=True)
  )
  (3): SeqPool()
  (4): MLP(
    (mlp): Sequential(
      (0): Linear(in_features=256, out_features=128, bias=True)
      (1): Tanh()
      (2): Linear(in_features=128, out_features=115, bias=True)
    )
  )
)
"""
``` 
## training_func
`training_func` implements function-based api for training neural networks, e.g., `train_one_epoch` and `train_multiple_epochs`.
### Strengths
This training tool can automatically adapt to your GPU devices by, e.g., DataParallel for multi-gpu training and gradiant accumulation for limited gpu memories.
### TODO
Implement `torch.utils.tensorboard` support for visualization.
```
train_one_epoch(
    model,
    training_args: TrainingArgumentsForLoop,
    dataset,
    compute_loss = default_feed,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler = None,
    global_step = 0,
    logger = None,
    show_bar = False
)
```
## trainer
`trainer` provides with a simplified `transformers.Trainer` class for multiple training controls, e.g., log, evaluate, checkpoint save and early stop.   
The function `train_multiple_epochs` are expected to support all these features of `trainer` in future.

## uitls
`utils` includes useful and compact api for I/O, pytorch, numpy ... It also has toolkits for logging and tokenization.  
For example, you can read a json dataset by
```
deepnlp.utils.read_json_line(filename, n_lines)
```
which returns a `List[dict]` dataset and can only read first `n_lines` during debug.

## data
`data` implements a torch Dataset class `UniversalDataset` supporting multiple types of input (list or dict samples).

# Test
## Trainer
To test the accelerate based trainer,
```
# for single gpu
python -m tests.test_acc_trainer

# launch with torchrun
torchrun --nproc-per-node=2 -m tests.test_acc_trainer

# launch with accelerate
accelerate launch --num_processes=2 -m tests.test_acc_trainer
```
Several settings to test:
- to test stopping on max_steps: `batch_size=16, device_batch_size=8, 2gpu, samples=100, max_steps=50, logging_steps=2`