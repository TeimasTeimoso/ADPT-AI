import torch
from torch import cuda
from torch.utils.data import DataLoader
import hydra
from hydra.core.config_store import ConfigStore
from transformers import DistilBertTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from confs.config import ADPTConfig
from dataloader.dataset import ADPTDataset
from model import ADPTModel
from dataloader import load_dataset
from train import train
from test import test
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

cs = ConfigStore.instance()
cs.store(name="adpt_config", node=ADPTConfig)


@hydra.main(version_base=None, config_path="confs", config_name="config")
def main(cfg: ADPTConfig):
    model = ADPTModel(20, [3], 1024, 128, 0.06)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.params.learning_rate)

    X_train, y_train = load_dataset(cfg.files.train_data)
    X_dev, y_dev = load_dataset(cfg.files.dev_data)
    X_test, y_test = load_dataset(cfg.files.test_data)

    X_train = pd.concat([X_train, X_dev]).reset_index(drop=True)
    y_train = pd.concat([y_train, y_dev]).reset_index(drop=True)
    y_train.replace(np.NaN, 0, inplace=True)

    X_train = X_train['text']
    X_test = X_test['text']

    train_set = ADPTDataset(X_train, y_train, tokenizer, cfg.params.max_len)
    test_set = ADPTDataset(X_test, y_test, tokenizer, cfg.params.max_len)


    params = {'batch_size': cfg.params.batch_size,
                'shuffle': True,
                'num_workers': 0}

    training_loader = DataLoader(train_set, **params)
    testing_loader = DataLoader(test_set, **params)

    train(model, optimizer, training_loader, cfg.params.epochs, device)
    
    test(model, testing_loader, device)
