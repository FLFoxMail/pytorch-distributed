import torch
# 导入 Dataloader
from torch.utils.data import DataLoader, Dataset
# 导入 F 函数
import torch.nn.functional as F
from datautils import MyTrainDataset
# 导入依赖 多线程
import torch.multiprocessing as mp
# 分布式采样器
from torch.utils.data.distributed import DistributedSampler
# 导入 DDP
from torch.nn.parallel import DistributedDataParallel as DDP
# 导入组处理函数
from torch.distributed import init_process_group, destroy_process_group
import os

# 分布式训练设置 rank 是分配给进程的唯一标识，world_size 是参与训练的进程总数
def ddp_setup(rank, world_size):
    # 设置环境变量, Master_addr 是主节点的 IP 地址，Master_port 是主节点的端口号
    os.environ['MASTER_ADDR'] = 'localhost'
    # 122355 
    os.environ['MASTER_PORT'] = '65535'
    torch.cuda.set_device(rank)
    print("初始化进程环境")
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
# 重新定义 Trainner 改动点在于，包装模型

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        # 改动点 1 device_ids 参数是一个列表，用于指定模型存在的 GPU 设备
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
    def _run_batch(self, souce, target):
        self.optimizer.zero_grad()
        output = self.model(souce)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()
        
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, target in self.train_data:
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self._run_batch(source, target)
            
    def _save_checkpoint(self, epoch):
        # 改动点 2 DDP 包装的 model 需要使用 module.state_dict() 获取模型参数
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
        
    def train(self, epochs: int):
        for epoch in range(epochs):
            self._run_epoch(epoch)
            # 改动点 3 DDP 保存模型时无需多少进程保存，只需要一个进程保存即可
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
def load_train_obj():
    train_set = MyTrainDataset(2048)
    print("数据加载完成")
    model = torch.nn.Linear(50, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    print("优化器加载完成")
    return train_set, model, optimizer

# 改动点4 修改 dataSampler，使用分布式采样器，用来确保在分布式环境，数据不会被重复采样
# 把 shuffle 参数设置为 False 的原因是，分布式采样器会自动进行 shuffle
def prepare_dataLoader(dataset, batch_size):    
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))

def start(rank: int, world_size: int, total_epochs: int, save_every: int, batch_size: int):
    # 改动点 5 在启动训练之前，需要先调用 ddp_setup 函数，设置多进程环境
    ddp_setup(rank, world_size)
    print("开始加载数据模型和优化器")
    data_set, model, optimizer = load_train_obj()
    train_data = prepare_dataLoader(data_set, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    
    # 改动点 6 训练结束后，需要调用 destroy_process_group 函数，销毁多进程环境
    destroy_process_group()
    
    # 配置参数并启动
if __name__ == "__main__":
    total_epochs = 5
    save_every = 2
    batch_size = 1
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")
    
    mp.spawn(start, args=(world_size, total_epochs, save_every, batch_size), nprocs=world_size)