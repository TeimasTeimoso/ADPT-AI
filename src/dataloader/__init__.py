from pathlib import Path
import pandas as pd
from typing import Tuple
from torch.utils.data import DataLoader
from dataloader.utils import split_dataframe
from dataloader.dataset import ADPTDataset

def load_dataset(ds_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_json(ds_path)

    return split_dataframe(df)


def create_data_loader(ds_path: Path, params: dict) -> DataLoader:
    X, y = load_dataset(ds_path)
    X = X['text']

    dataset = ADPTDataset(X, y, params.pop('tokenizer'), params.pop('max_len'))

    return DataLoader(dataset, **params)