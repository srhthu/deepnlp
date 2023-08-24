# Distributed training
## Startup of Distributed Training
We introduce how to start a distributed training and the configuration process.
### Lunch
The multiple processes are lunched with related variables, e.g., local rank, which are passed through two ways:  
- `torch.distributed.launch` pass variables through command line arguments
- `torchrun` pass through environment variables

These variables related to mp are (part):
- `LOCAL_RANK` - The local rank
- `RANK` The global rank.
- `LOCAL_WORLD_SIZE` - The local world size of the node; equals to --nproc-per-node specified on torchrun.
- `WORLD_SIZE` - Total number of workers(processes).
- `MASTER_ADDR` - The FQDN of the host that is running worker with rank 0; used to initialize the Torch Distributed backend.
- `MASTER_PORT` - The port on the MASTER_ADDR that can be used to host the C10d TCP store.

### Init of Each Progress
The aim of initialization is to setup the communication between processes.  
`dist.init_process_group` is to initialize each process. By default, it uses *gloo* and *nccl* backend, and init through environment variables with `init_method="env://"`.

#### Initialization Methods
- TCP
    - should provide the IP and port of rank 0 process and each process's rank and world_size
- Shared file-system
- Environment variable (default, good for torchrun)
    - Four env vars are need: MASTER_PORT, MASTER_ADDR, WORLD_SIZE, RANK

*Note*: for initialization, local rank and local world size is not need as the communication do not discriminate nodes. However, handling GPU allocation requires local rank.

### GPU Allocation
Currently, each process can access all available GPUs (or gpus indicated by CUDA_VISIBLE_DEVICES). The model should be explicitly move to the correspond GPU of a process.
```Python
import os
model.to(int(os.environs["LOCAL_RANK"]))
```
Use `LOCAL_RANK` is critical if there are more than one nodes.

### Issues
#### gloo backend connection refused
If only initialize the nccl backend with two nodes, it works
```Bash
# node 1
torchrun --nproc_per_node=2 --master_addr 172.28.176.19 --master_port 15432 --nnodes 2 --node_rank 0 start.py
# node 2
torchrun --nproc_per_node=2 --master_addr 172.28.176.19 --master_port 15432 --nnodes 2 --node_rank 1 start.py
```
However, if initialize with gloo backend, either by default or explicitly pass the argumetn. The error message of each process looks like:
```
RuntimeError connect [127.0.1.1]:<port>: Connection refused
```
the port is different for each process.

But `gloo` works for single node with multiple workers.

Note: `nccl` is optimized for CUDA tensors. Recommended for CUDA collective operations.

## Communication
We introduce the collective communication here, while there are also other communication types e.g. Point-to-Point but rarely accessed in training for general users.

Some functions:
- reduce
- all_reduce
- gather
- all_gather

See demos in `start.py`

As noted in `ddp_tutorial`, "The collective communication functions, e.g., all_reduce, serve as a synchronization.". Which means that after calling these functions all progresses are synchronized.