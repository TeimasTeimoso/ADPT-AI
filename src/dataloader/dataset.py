import torch
from torch.utils.data import Dataset
import numpy as np

class ADPTDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = X
        self.labels = np.array(y.values.tolist())
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.tokenizer.tokenize(self.text[index])
        text = list(map(lambda t: tokenizer.sep_token if t == '\n' else t, text))

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']


        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.float)
        }
     