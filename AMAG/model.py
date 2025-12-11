import torch
import torch.nn as nn
from SI import SI
from TE import TE
from TR import TR

class AMAG(nn.Module):
    def __init__(self, config):
        super(AMAG, self).__init__()
        # 初始化
        self.C = config['num_channels']
        self.D = config['input_dim']
        self.d = config['hidden_dim']
        self.pred_len = config['pred_len']
        self.mode = config['mode']
         
        self.te = TE(self.D, self.d)
        self.si = SI(self.C, self.d)
        self.tr = TR(self.d, self.D) 
        
    def initialize_adjacency(self, data_loader, device):
        """
        initialize_adjacency 的 Docstring
        初始化邻接矩阵 A_a 和 A_m
        """
        print("Initializing Adjacency Matrices with Correlation...")
        batch_x, _ = next(iter(data_loader)) # (B, T, C, D)
        batch_x = batch_x.to(device).squeeze(-1) # (B, T, C)
        
        # Permute to (B, C, T) -> Flatten to (C, B*T)
        data_flat = batch_x.permute(0, 2, 1).reshape(self.C, -1)
        
        corr_matrix = torch.corrcoef(data_flat)
        corr_matrix = torch.nan_to_num(corr_matrix, 0.0)
        
        # 赋值给A_a 和 A_m
        with torch.no_grad():
            self.si.A_a.data.copy_(corr_matrix)
            self.si.A_m.data.copy_(corr_matrix)
        print("Initialization Complete.")

    def forward(self, x):
        B, T, C, D = x.shape
        
        # 初始化隐状态
        h_te = None 
        h_tr = None 
        
        predictions = []
        
        curr_x = None
        
        for t in range(T):
            curr_x = x[:, t, :, :] # (B, C, D)
            # TE 
            h_emb, h_te = self.te(curr_x, h_te)
            # SI
            z_t = self.si(h_emb)
            # TR
            pred_x, h_tr = self.tr(z_t, h_tr)
        
        predictions.append(pred_x)
        
        if self.mode == 'multi_step': 
            # 使用上一步的预测作为下一步的输入
            current_input = pred_x
            
            for _ in range(self.pred_len - 1):
                h_emb, h_te = self.te(current_input, h_te)
                z_t = self.si(h_emb)
                next_pred, h_tr = self.tr(z_t, h_tr)
                
                predictions.append(next_pred)
                current_input = next_pred
                
        # Stack predictions: (Batch, Pred_Len, C, D)
        return torch.stack(predictions, dim=1)