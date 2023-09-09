import torch
from torch.utils.data import WeightedRandomSampler

def inverse_class_weight(targets):
    class_weights = torch.bincount(targets)
    return 1.0/class_weights

def weighted_sampler(dataset):
    targets = torch.tensor(dataset.labels, dtype=torch.int64)
    targets = torch.sum(targets, dim=0)
    computed_weights = inverse_class_weight(targets)
    print(computed_weights)
    return WeightedRandomSampler(weights=computed_weights, num_samples=len(dataset))