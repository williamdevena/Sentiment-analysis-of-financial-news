import pandas as pd

from utils import costants


class FinancialNewsDataset():

    def __init__(self, path_csv) -> None:
        self.path_csv = path_csv
        self.data = pd.read_csv(self.path_csv, sep=",")
        self.text = self.data['text']
        self.sentiment = self.data['sentiment']
        #print(self.data, self.text, self.sentiment)
        #print(self.text[0], self.sentiment[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.text[idx]
        sentiment = self.sentiment[idx]

        return text, sentiment
