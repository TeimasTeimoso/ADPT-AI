from tqdm import tqdm
import torch
from utils.loss import AsymmetricLoss

loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=0)

def train(model, optimizer, training_loader, epoch, device):
  running_loss = 0.0
  pos_loss = 0.0
  neg_loss = 0.0
  model.train()
  for _,data in tqdm(enumerate(training_loader, 0)):
      ids = data['input_ids'].to(device, dtype = torch.long)
      mask = data['attention_mask'].to(device, dtype = torch.long)
      token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
      targets = data['labels'].to(device, dtype = torch.float)

      pos_mask = (targets==1)
      neg_mask = (targets==0)

      outputs = model(ids, mask, token_type_ids, epoch)

      optimizer.zero_grad()
      loss = loss_fn(outputs, targets)
      pos_loss = loss_fn(outputs[pos_mask], targets[pos_mask])
      neg_loss = loss_fn(outputs[neg_mask], targets[neg_mask])

      loss.backward()

      running_loss += loss.item()
      pos_loss += pos_loss.item()
      neg_loss += neg_loss.item()
        
      optimizer.step()

  epoch_loss = running_loss / len(training_loader)
  pos_loss = pos_loss / len(training_loader)
  neg_loss = neg_loss / len(training_loader)

  print("Epoch {} - Loss: {:.4f}".format(epoch + 1, epoch_loss))
  return epoch_loss, pos_loss, neg_loss
