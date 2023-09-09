import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
from typing import Tuple

def load_dataset(ds_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_json(ds_path)

    return split_dataframe(df)

def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  y_raw = df['labels']

  y = convert_labels_list_to_columns(y_raw)
  X = df.drop(columns=['labels'])

  return X, y

def convert_labels_list_to_columns(labels: pd.Series) -> pd.DataFrame:
  mlb = MultiLabelBinarizer()
  encoded_labels = mlb.fit_transform(labels)

  return pd.DataFrame(encoded_labels, columns=mlb.classes_)