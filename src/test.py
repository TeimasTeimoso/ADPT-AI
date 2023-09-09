import torch
from torch import cuda
from sklearn.metrics import f1_score
import numpy as np

def test_epoch(model, test_loader, loss_fn):
    device = 'cuda' if cuda.is_available() else 'cpu'

    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.float)
            
            outputs = model(ids, mask)

            loss = loss_fn(outputs, targets)

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            
            y_true.append(targets.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())
    
            running_loss += loss.item()

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    epoch_loss = running_loss / len(test_loader)

    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=1)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=1)
    f1_table = f1_score(y_true, y_pred, average=None, zero_division=1)

    return epoch_loss, f1_micro, f1_macro, f1_table