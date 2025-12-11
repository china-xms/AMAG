import torch
import torch.nn as nn

class TE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TE, self).__init__()
        """
        Temporal Encoding Module 
        使用 GRU 独立编码每个通道
        """
        # 输入维度 D, 隐层维度 d
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, x, h_prev=None):
        """
        x: (Batch, Num_Channels, Input_Dim) -> 需要 reshape 成 (Batch*C, 1, D) 用于单步
           或者 (Batch, Time, C, D) 处理整个序列
        
        这里实现单步更新逻辑，以便支持递归预测
        Input x: (Batch, Num_Channels, Input_Dim)
        h_prev: (1, Batch*Num_Channels, Hidden_Dim)
        """
        B, C, D = x.shape
        
        # Reshape 为 (Batch * Num_Channels, 1, Input_Dim) 以共享权重处理所有通道
        x_reshaped = x.reshape(B * C, 1, D)
        
        # GRU Forward
        # out: (B*C, 1, Hidden_Dim), h_new: (1, B*C, Hidden_Dim)
        out, h_new = self.gru(x_reshaped, h_prev)
        
        # 还原形状: (Batch, Num_Channels, Hidden_Dim)
        embedding = out.reshape(B, C, -1)
        
        return embedding, h_new