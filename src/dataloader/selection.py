from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from skmultilearn.model_selection import IterativeStratification

kfold = IterativeStratification(n_splits=5)

def split_into_folds(dataset, loader_config):
    loader_folds = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset.text, dataset.labels)):
        train_subsampler = SubsetRandomSampler(train_idx)
        test_subsampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, sampler=train_subsampler, **loader_config)
        test_loader = DataLoader(dataset, sampler=test_subsampler, **loader_config)

        loader_folds.append((train_loader, test_loader))

    return loader_folds