#!/bin/bash
# no tail drops
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.01 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.02 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.03 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.04 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.05 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.06 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.07 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.08 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.09 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.1 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 1 -hd 0 -nr 0 -n 2 --epochs 90 -sp "AWS" -cm "nccl"
# tail drops
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.01 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.02 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.03 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.04 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.05 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.06 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.07 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.08 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.09 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 0.1 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.17.0.2" --master_port=12355 toy-ddp.py -rn 1 -dc 1 -nr 0 -n 2 --epochs 90 -td 1 -hd 0 -sp "AWS" -cm "nccl"