from pathlib import Path
import pandas as pd
from typing import Tuple
from dataloader.utils import split_dataframe

def load_dataset(ds_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_json(ds_path)

    return split_dataframe(df)