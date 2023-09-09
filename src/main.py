import torch
from torch import cuda
from torch.utils.data import DataLoader
import hydra
from hydra.core.config_store import ConfigStore
from transformers import DistilBertTokenizer
import pandas as pd
import numpy as np

from dataloader.dataset import ADPTDataset
from model import ADPTModel
from dataloader import load_dataset
from dataloader.selection import split_into_folds
from train import train_epoch
from test import test_epoch
from utils.loss import AsymmetricLoss

import ray
from ray import  tune
from ray.tune.search.hyperopt import HyperOptSearch

device = 'cuda' if cuda.is_available() else 'cpu'

train_data_path = '/app/data/task 1/training_set_task1.json'
dev_data_path = '/app/data/task 1/dev_set_task1.json'
test_data_path = '/app/data/task 1/test_set_task1.json'

def import_data(train, dev, tokenizer):
    params = {'batch_size': 8,
                'num_workers': 0,
                'max_len': 128,
                'tokenizer': tokenizer
    }

    X_train, y_train = load_dataset(train)
    X_dev, y_dev = load_dataset(dev)

    X_train = pd.concat([X_train, X_dev]).reset_index(drop=True)
    y_train = pd.concat([y_train, y_dev]).reset_index(drop=True)
    y_train.replace(np.NaN, 0, inplace=True)

    X_train = X_train['text']

    dataset = ADPTDataset(X_train, y_train, params.pop('tokenizer'), params.pop('max_len'))

    folds = split_into_folds(dataset, params)

    return folds

def objective(config):
    step = 1

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    loss_fn = AsymmetricLoss()

    folds = import_data(train_data_path, dev_data_path, tokenizer)

    history = {'val_loss': [],'f1_micro': [], 'f1_macro': [], 'f1_table': []}

    for fold, (training_loader, testing_loader) in enumerate(folds):
        model = ADPTModel(20, config['kernels'], config['hidden_dim'], int(config['filters']), config['dropout'])
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['l2'])

        for epoch in range(10):
            train_epoch(model, optimizer, training_loader, loss_fn, epoch)
        
        epoch_loss, f1_micro, f1_macro, f1_table = test_epoch(model, testing_loader, loss_fn)

        print(f"F1 Micro: {f1_micro}")
        print(f"F1 Macro: {f1_macro}")
        print(f"F1 Table: {f1_table}")

        history['f1_micro'].append(f1_micro)
        history['f1_macro'].append(f1_macro)
        history['f1_table'].append(f1_table)
        history['val_loss'].append(epoch_loss)

    avg_f1_micro = np.mean(history['f1_micro'])
    avg_f1_macro = np.mean(history['f1_macro'])
    avg_f1_table = np.mean(history['f1_table'], axis=0)
    avg_val_loss = np.mean(history['val_loss'])

    print(f'Average F1 Macro: {avg_f1_table}')

    return {"f1_micro": avg_f1_micro, "f1_macro": avg_f1_macro, 'val_loss': avg_val_loss}

if __name__ == "__main__":

    config = {
            "lr": tune.loguniform(1e-6, 1e-4),
            "dropout": tune.loguniform(0.05, 0.45),
            "kernels": tune.choice([[1,2], [3], [3,4,5]]),
            "l2": tune.loguniform(0.01, 0.2),
            "filters": tune.choice([2**i for i in range(5, 8)]),
            "hidden_dim": tune.choice([2**i for i in range(8, 11)])
    }

    num_samples = 10

    ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
    ray.init()

    tuner = tune.Tuner(
    tune.with_resources(objective, {"gpu": 1}),
    tune_config=tune.TuneConfig(
        search_alg=HyperOptSearch(metric="f1_micro", mode="max"),
        metric="f1_micro",
        mode="max",
        num_samples=num_samples,
    ),
    param_space=config,
    )   

    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)