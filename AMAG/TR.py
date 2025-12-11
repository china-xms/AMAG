import torch
import torch.nn as nn

class TR(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(TR, self).__init__()
        """
        TR Module 
        """
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z_t, h_prev=None):
        """
        Input z_t: (Batch, Num_Channels, Hidden_Dim)
        """
        B, C, D = z_t.shape
        
        # Reshape: (Batch * C, 1, Hidden_Dim)
        z_reshaped = z_t.reshape(B * C, 1, D)
        
        # GRU
        out, h_new = self.gru(z_reshaped, h_prev)
        
        # FC 
        # (Batch * C, 1, Output_Dim)
        pred = self.fc(out)
        
        # Restore shape: (Batch, Num_Channels, Output_Dim)
        pred = pred.reshape(B, C, -1)
        
        return pred, h_new