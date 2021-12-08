#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.0 -nr 0 -n 2 --epochs 10 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.01 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.02 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.03 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.04 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.05 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.06 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.07 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.08 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.09 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 0.1 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=12355 toy-ddp.py -rn 1 -dc 1 -nr 0 -n 2 --epochs 90 -td 1 -hd 1 -sp "AWS" -cm "nccl"