import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MultiLabelBinarizer

def split_into_tokens(text) -> str:
    return " ".join(text.split())

def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  labels_col = df.pop('labels')

  y = convert_labels_list_to_columns(labels_col)

  return df, y

def convert_labels_list_to_columns(labels: pd.Series) -> pd.DataFrame:
  mlb = MultiLabelBinarizer()
  encoded_labels = mlb.fit_transform(labels)

  return pd.DataFrame(encoded_labels, columns=mlb.classes_)