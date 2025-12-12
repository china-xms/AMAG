import torch
import torch.nn as nn
import torch.nn.functional as F

class SI(nn.Module):
    def __init__(self, num_channels, hidden_dim):
        super(SI, self).__init__()
        """
        Spatial Interaction Module 
        """
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # 可学习的邻接矩阵 A_a 和 A_m 
        self.A_a = nn.Parameter(torch.zeros(num_channels, num_channels))
        self.A_m = nn.Parameter(torch.zeros(num_channels, num_channels))

        # MLP
        self.adaptor_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # FC layes 
        self.fc_a = nn.Linear(hidden_dim, hidden_dim)
        self.fc_m = nn.Linear(hidden_dim, hidden_dim)

        # beta 参数 
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.beta2 = nn.Parameter(torch.tensor(1.0))
        self.beta3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_t):
        """
        h_t shape: (Batch, Num_Channels, Hidden_Dim)
        """
        B, C, D = h_t.shape
        
        # Add Module
        h_u = h_t.unsqueeze(2).expand(-1, -1, C, -1) # Source (u)
        h_v = h_t.unsqueeze(1).expand(-1, C, -1, -1) # Target (v)
        h_pair = torch.cat([h_u, h_v], dim=-1)
        
        # S: (B, C, C, 1) -> (B, C, C)
        S = self.adaptor_mlp(h_pair).squeeze(-1)
        # Add Message Passing: a_v = sum(S_uv * A_uv * h_u) 
        A_a_batch = self.A_a.unsqueeze(0).expand(B, -1, -1)
        # 权重矩阵 W_add = S * A_a
        W_add = S * A_a_batch
        a_t = torch.bmm(W_add.transpose(1, 2), h_t)

        # Modulator Module
        # m_uv = A_m_uv * (h_u * h_v) (Hadamard product) 
        hadamard = h_t.unsqueeze(2) * h_t.unsqueeze(1) 
        A_m_expanded = self.A_m.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D)
        m_t = torch.sum(A_m_expanded * hadamard, dim=1)

        # Fusion
        # z_t = b1*h_t + b2*FC(a_t) + b3*FC(m_t)
        z_t = self.beta1 * h_t + \
              self.beta2 * self.fc_a(a_t) + \
              self.beta3 * self.fc_m(m_t)
              
        return z_t