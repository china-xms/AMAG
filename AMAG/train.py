import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

def train_model(model, train_loader, optimizer, device, epochs):
    criterion = nn.MSELoss()  
    model.train()
    time_all_start = time.time()
    for epoch in range(epochs):
        time_start = time.time()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            # x: (B, T, C, D), y: (B, Pred_Len, C, D)
            
            optimizer.zero_grad()
            
            preds = model(x) # (B, Pred_Len, C, D)
            
            # 确保预测长度和标签长度一致 
            if preds.shape[1] != y.shape[1]:
                y = y[:, :preds.shape[1], :, :]
            
            loss = criterion(preds, y)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        time_end = time.time()
        print(f"Epoch {epoch+1} completed in {time_end - time_start:.2f} seconds.")
        print(f"Epoch {epoch+1}/{epochs} Avg Loss: {total_loss / len(train_loader):.6f}")
    print(f"Total Training Time: {time.time() - time_all_start:.2f} seconds.")
    
    return model