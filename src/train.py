from tqdm import tqdm
import torch
from torch import cuda
from sklearn.metrics import f1_score

from ray.air import session

def train_epoch(model, optimizer, train_loader, loss_fn, epoch):
  device = 'cuda' if cuda.is_available() else 'cpu'

  running_loss = 0.0
  model.train()
  for _,data in tqdm(enumerate(train_loader, 0)):
    ids = data['input_ids'].to(device, dtype = torch.long)
    mask = data['attention_mask'].to(device, dtype = torch.long)
    targets = data['labels'].to(device, dtype = torch.float)

    optimizer.zero_grad()

    outputs = model(ids, mask)
    loss = loss_fn(outputs, targets)
    
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  epoch_loss = running_loss / len(train_loader)
  print("Training Epoch {} - Loss: {:.4f}".format(epoch + 1, epoch_loss))

  return epoch_loss