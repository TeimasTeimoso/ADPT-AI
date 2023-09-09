import torch
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
from utils.loss import AsymmetricLoss

loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=0)

def test(model, testing_loader, device):
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(testing_loader):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.float)
            
            outputs = model(ids, mask, token_type_ids, 0)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            
            y_true.append(targets.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print("Micro F1-Score: {:.4f}".format(f1_score(y_true, y_pred, average='micro', zero_division=0)))
    print("Macro F1-Score: {:.4f}".format(f1_score(y_true, y_pred, average='macro', zero_division=0)))
    print(f"F1-score matrix: {f1_score(y_true, y_pred, average=None, zero_division=0)}")
    recall = recall_score(y_true, y_pred, average=None,  zero_division=0)
    print("Recall: ", recall),
    precision = precision_score(y_true, y_pred, average=None,  zero_division=0)
    print("Precision: ", precision)


def evaluate(model, loader, device, epoch):
    model.eval()

    y_true = []
    y_pred = []
    pos_loss = 0.0
    neg_loss = 0.0
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.float)
            
            pos_mask = (targets==1)
            neg_mask = (targets==0)

            outputs = model(ids, mask, token_type_ids, epoch)
            
            loss = loss_fn(outputs, targets)
            pos_loss = loss_fn(outputs[pos_mask], targets[pos_mask])
            neg_loss = loss_fn(outputs[neg_mask], targets[neg_mask])

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            
            y_true.append(targets.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())

            running_loss += loss.item()
            pos_loss += pos_loss.item()
            neg_loss += neg_loss.item()

        epoch_loss = running_loss / len(loader)
        pos_loss = pos_loss / len(loader)
        neg_loss = neg_loss / len(loader)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return (epoch_loss, pos_loss.cpu(), neg_loss.cpu(), micro_f1, macro_f1)




