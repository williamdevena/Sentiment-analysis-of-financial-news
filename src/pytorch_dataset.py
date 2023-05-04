import pandas as pd

from src import data_processing
from utils import costants


class FinancialNewsDataset():
    def __init__(self, path_csv, dict_ds):
        self.path_csv = path_csv
        self.data = pd.read_csv(self.path_csv, sep=",")
        self.text = self.data['text']
        self.sentiment = self.data['sentiment']
        list_tokenized_sentences, max_len_sent = data_processing.create_list_tokenized_words(data=self.data,
                                                                               dict=dict_ds)
        list_int_sentences = data_processing.create_list_encoded_words(dict_ds, list_tokenized_sentences)
        self.list_int_sentences_padded = data_processing.pad_sentences(list_int_sentences, max_len_sent+3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.list_int_sentences_padded[idx]
        ## from -1,0,+1 to 0,1,2
        sentiment = self.sentiment[idx]+1


        return text, sentiment