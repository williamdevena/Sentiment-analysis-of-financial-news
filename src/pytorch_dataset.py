import pandas as pd
import torch

from src import data_processing
from typing import Dict


class FinancialNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)




