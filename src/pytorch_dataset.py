import pandas as pd
import torch

from src import data_processing


class FinancialNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# class FinancialNewsDataset():
#     def __init__(self, path_csv, words_dict, max_length_for_padding):
#         self.path_csv = path_csv
#         self.max_length_for_padding = max_length_for_padding
#         self.words_dict = words_dict
#         # self.data = pd.read_csv(self.path_csv, sep=",")
#         # self.text = self.data['text']
#         # self.sentiment = self.data['sentiment']
#         # list_tokenized_sentences, max_len_sent = data_processing.create_list_tokenized_words(data=self.data,
#         #                                                                        dict=dict_ds)
#         # list_int_sentences = data_processing.create_list_encoded_words(dict_ds, list_tokenized_sentences)
#         # self.list_int_sentences_padded = data_processing.pad_sentences(list_int_sentences, max_len_sent+3)
#         self.list_embeddings, self.sentiment = data_processing.vectorize_ds(path_csv=self.path_csv,
#                                                                 words_dict=self.words_dict)

#         self.vectorized_ds = data_processing.pad_list_embeddings(list_embeddings=self.list_embeddings,
#                                                             desired_sent_length=self.max_length_for_padding)

#     def __len__(self):
#         return self.vectorized_ds.shape[0]

#     def __getitem__(self, idx):
#         # text = self.list_int_sentences_padded[idx]
#         # ## from -1,0,+1 to 0,1,2
#         # sentiment = self.sentiment[idx]+1
#         text = self.vectorized_ds[idx]
#         sentiment = self.sentiment[idx]


#         return text, sentiment




