import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

class ADPTModel(nn.Module):
  def __init__(self, num_classes, kernel_sizes, hidden_dim=768, num_filters=256, dropout_rate=0.05):
    super(ADPTModel, self).__init__()

    self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    for name, param in self.distilbert.named_parameters():
      if name.startswith('transformer.layer.0') or name.startswith('transformer.layer.1'):
          param.requires_grad = False

    self.l0 = nn.Linear(768, hidden_dim)
    self.d0 = nn.Dropout(dropout_rate)

    self.convs = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim, out_channels=num_filters, kernel_size=ks) for ks in kernel_sizes])

    self.l1 = nn.Linear(len(kernel_sizes) * num_filters, hidden_dim)

    self.d1 = nn.Dropout(dropout_rate)
    
    self.fc = nn.Linear(hidden_dim, num_classes)

  def forward(self, input_ids, attention_mask):
    x = self.distilbert(input_ids, attention_mask)
    hidden_state = x[0]
    hidden_state = F.elu(self.l0(hidden_state))
    hidden_state = self.d0(hidden_state)
    
    hidden_state = hidden_state.permute(0, 2, 1) # (batch_size, embedding_dim, seq_len)

    x_convs = [F.relu(conv(hidden_state)) for conv in self.convs] # (kernel_len x (batch_size, num_filters, out_channels))
    x_pools = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x_convs] # (kernel_len x (batch_size, num_filters))
    x_cat = torch.cat(x_pools, dim=1) # (batch_size x (kernel_len*num_filters))

    x_out = F.elu(self.l1(x_cat))
    x_out = self.d1(x_out)
    x_out = self.fc(x_out)

    return x_out