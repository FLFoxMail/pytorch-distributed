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
def ddp_setup():
    print("初始化进程环境")
    # 相比于 ddp_setup(rank, world_size)， 使用 torchrun 启动多进程时，由 torchrun 自动分配 rank 和 world_size
    # nccl 是一种用于 GPU 之间通信的库，在多 GPU 训练中，nccl 是默认的通信后端， nccl 的含义是 
    # NVIDIA Collective Communications Library 即 NVIDIA 集合通信库
    # 除了 nccl 还有 gloo 和 mpi，gloo 是一种通用的通信后端， mpi 是一种用于分布式计算的通信协议 nccl 在windows 上不可用
    init_process_group(backend='gloo')

    
# 重新定义 Trainner 改动点在于，包装模型

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int, 
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        # 更新快照
        self.epoch_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        # 改动点 1 device_ids 参数是一个列表，用于指定模型存在的 GPU 设备
        self.model = DDP(self.model, device_ids=[self.gpu_id])
    
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epoch_run = snapshot["EPOCH_RUN"]
        print(f"Resume training from epoch {self.epoch_run}")
        
        
    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCH_RUN"] = epoch
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        
        
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

        
    def train(self, epochs: int):
        for epoch in range(self.epoch_run, epochs):
            self._run_epoch(epoch)
            # 改动点 3 DDP 保存模型时无需多少进程保存，只需要一个进程保存即可
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                
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

def start(total_epochs: int, save_every: int, batch_size: int, snapshot_path = "snapshot.pt"):
    ddp_setup()
    print("开始加载数据模型和优化器")
    data_set, model, optimizer = load_train_obj()
    train_data = prepare_dataLoader(data_set, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    
    # 配置参数并启动
if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    start(total_epochs, save_every, batch_size)
