# 文件：torch_test.py
import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    print(f"Rank {rank} on {os.uname().nodename}: CUDA可用 {torch.cuda.is_available()}, 设备数 {torch.cuda.device_count()}")

if __name__ == "__main__":
    main()