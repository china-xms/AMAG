import torch
import torch.nn as nn
import time
def test_model(model, test_loader, device):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0
    total_steps = 0
    time_start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            if preds.shape[1] != y.shape[1]:
                y = y[:, :preds.shape[1], :, :]
                
            loss = criterion(preds, y)
            total_loss += loss.item()
            total_steps += 1
            
    avg_loss = total_loss / total_steps
    time_end = time.time()
    print(f"Testing Time: {time_end - time_start:.2f} seconds\n")
    print(f"Test MSE Loss: {avg_loss:.6f}")
    return avg_loss