from dataclasses import dataclass

@dataclass
class Files:
    train_data: str
    dev_data: str
    test_data: str

@dataclass
class Params:
    max_len: int
    batch_size: int
    learning_rate: float
    epochs: int

@dataclass
class ADPTConfig:
    files: Files
    params: Params