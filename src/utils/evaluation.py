import pandas as pd
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm # progress bar
from typing import List
import time

def evaluate_pipeline_x_validation(pipeline: Pipeline, pipeline_grid: ParameterGrid, models_name: List[str], X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame: 
  pipeline_scores = pd.DataFrame(columns=['model', 'f1-micro'])

  for model, params in tqdm(zip(models_name, pipeline_grid), total=len(models_name)):
    pipeline.set_params(**params)
    score = cross_val_score(pipeline, X, Y, cv=10, scoring='f1_micro')
    pipeline_scores = pd.concat([pipeline_scores, pd.DataFrame({'model': model, 'f1-micro': score.mean()}, index=[0])], ignore_index=True)

  return pipeline_scores


def evaluate_model(model, X_train, y_train, X_test, y_test, pipeline, name="Model"):

    start = time.time()
    pipeline.set_params(**model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    end = time.time()

    print(f"Micro F1-score for {name} is {f1_score(y_test, y_pred, average='micro', zero_division=0)}")
    print(f"Macro F1-score for {name} is {f1_score(y_test, y_pred, average='macro', zero_division=0)}")
    print(f"Time to train the {name}: {end - start} seconds")
    print(f"F1-Score for {name} is {f1_score(y_test, y_pred, average=None, zero_division=0)}")
    print(f"Precision for {name} is {precision_score(y_test, y_pred, average=None, zero_division=0)}")
    print(f"Recall for {name} is {recall_score(y_test, y_pred, average=None, zero_division=0)}")
    print("\n")