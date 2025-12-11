import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self, num_samples, num_channels, seq_len, pred_len, connection_ratio=0.2):
        """
        X_t = X_{t-1} + G(X_{t-1}, A) + noise
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.Layer_norm = torch.nn.LayerNorm(num_channels)
        
        with torch.no_grad():
            # 生成邻接矩阵 A
            self.adj = np.random.rand(num_channels, num_channels)
            mask = np.random.rand(num_channels, num_channels) < connection_ratio
            self.adj = self.adj * mask
            # 保证特征值 < 1
            eig_vals = np.linalg.eigvals(self.adj)
            max_eig = np.max(np.abs(eig_vals))
            if max_eig > 1:
                self.adj = self.adj / (1.1 * max_eig)
        
            self.adj = torch.FloatTensor(self.adj)
        
            # 生成数据
            self.data = []
            for _ in range(num_samples):
                # 随机初始化 X_0
                x = torch.randn(num_channels, 1) 
                sequence = [x]
            
                for _ in range(self.total_len - 1):
                    noise = torch.randn(num_channels, 1) * 0.1
                    x_next = torch.mm(self.adj, x) + noise + x
                    x_next = self.Layer_norm(x_next.T).T 
                    sequence.append(x_next)
                    x = x_next
            
                # Stack: (Total_Len, C, D)
                sample_data = torch.stack(sequence).squeeze(-1)
                self.data.append(sample_data.detach()) # D=1

            self.data = torch.stack(self.data).unsqueeze(-1) # (N, Total_Len, C, D)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 对人工数据集切分
        # src: [0 : T]
        # tgt: [T : T+tau] (Multi-step) or [1 : T+1] (One-step)
        sample = self.data[idx]
        src = sample[:self.seq_len]
        tgt = sample[self.seq_len : self.seq_len + self.pred_len]
        return src, tgt