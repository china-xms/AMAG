import yaml
import torch
from torch.utils.data import DataLoader, random_split
from dataset import SyntheticDataset
from model import AMAG
from train import train_model
from test import test_model
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 读取配置
    config = load_config('config.yaml')
    print(f"Loaded config: {config}")
    
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 准备数据集 
    full_dataset = SyntheticDataset(
        num_samples=config['total_samples'],
        num_channels=config['num_channels'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'] if config['mode'] == 'multi_step' else 1
    )
    
    # 划分训练集和测试集 (70% / 30%)
    train_size = int(config['train_ratio'] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # 初始化模型
    model = AMAG(config).to(device)
    
    # 初始化邻接矩阵 
    model.initialize_adjacency(train_loader, device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # 训练
    print("Starting Training...")
    model = train_model(model, train_loader, optimizer, device, config['epochs'])
    # 测试
    print("Starting Testing...")
    mse = test_model(model, test_loader, device)
    
    print("Done.")


if __name__ == "__main__":    
    main()
