import pandas as pd

#######################
# Statistic Functions
#######################
def count_class_entries(pd: pd.DataFrame) -> pd.DataFrame:
  entries = pd.shape[0]
  class_count = pd.sum(axis=0).sort_values(ascending=False)
  normalized_class_count = class_count / entries
  normalized_class_count = normalized_class_count.to_frame()
  return normalized_class_count

####################
# labels normalization
####################
def set_missing_labels(original_y, labels_list):
    """Set missing labels to 0"""

    missing_labels = labels_list.difference(original_y.columns)
    for label in missing_labels:
        original_y[label] = 0
    
    return original_y