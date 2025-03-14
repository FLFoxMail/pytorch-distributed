# NOTE: Redirects are currently not supported in Windows or MacOs.
# [W socket.cpp:69#7] [c10d] The client socket has failed to connect to [::ffff:8.134.52.142]:40357 (system error: 10049 - 在其上下文中，该请求的地址无效。).

# 解决报错问题测试 socket连接
import torch.distributed as dist
import torch

def test_connection():
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.all_reduce(torch.tensor([1.0]), op=dist.ReduceOp.SUM)
    dist.destroy_process_group()

if __name__ == '__main__':
    test_connection()