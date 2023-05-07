import logging
from copy import deepcopy

import nltk
import numpy as np
import pandas as pd
import torch
import torchtext
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils import costants


def build_dict(filename):
    words = dict()
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            try:
                words[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue
    logging.info("- DICTIONARY BUILT")
    return words


def message_to_token_list(sent, words_dict):
    lemmatizer = WordNetLemmatizer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(sent)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words_dict]

    return useful_tokens


def message_to_word_vectors(sent, words_dict):
    processed_list_of_tokens = message_to_token_list(sent=sent,
                                                     words_dict=words_dict)
    vectors = []
    for token in processed_list_of_tokens:
        if token not in words_dict:
            continue
        token_vector = words_dict[token]
        vectors.append(token_vector)

    return np.array(vectors, dtype=float)



def vectorize_ds(path_csv, words_dict):
    data = pd.read_csv(path_csv, sep=",")
    text = data['text']
    sentiment = data['sentiment']

    vectorized_text = []
    for sent in text:
        vectorized_sent = message_to_word_vectors(sent=sent,
                                                  words_dict=words_dict)
        vectorized_text.append(vectorized_sent)

    logging.info(f"- VECTORIZED DATASET IN {path_csv}")

    return vectorized_text, np.array(sentiment)




def pad_list_embeddings(list_embeddings, desired_sent_length):
  list_embeddings_padded = deepcopy(list_embeddings)

  for i, sent in enumerate(list_embeddings):
    x_seq_len = sent.shape[0]
    sent_length_difference = desired_sent_length - x_seq_len

    pad = np.zeros(shape=(sent_length_difference, 50))

    list_embeddings_padded[i] = np.concatenate([sent, pad])

  return np.array(list_embeddings_padded).astype(float)









# def custom_tokenizer(text):
#     print(text)
#     return nltk.word_tokenize(text)


# def dataset_preparation():
#     ### Defining the feature processing
#     TEXT = torchtext.data.Field(
#         tokenize=custom_tokenizer, # default splits on whitespace
#         #tokenizer_language='en_core_web_sm'
#     )
#     ### Defining the label processing
#     LABEL = torchtext.data.LabelField(dtype=torch.long)

#     fields = [('text', TEXT), ('sentiment', LABEL)]

#     dataset = torchtext.data.TabularDataset(
#         path=costants.CSV_PATH,
#         format='tsv',
#         skip_header=False,
#         fields=fields,
#         #csv_reader_parameters={'sep':"\t"}
#         )

#     return dataset






# def pad_sentences(list_int_sentences, max_seq_length):
#     list_int_sentences_padded = np.zeros((len(list_int_sentences), max_seq_length), dtype = int)

#     for i, sent in enumerate(list_int_sentences):
#         sent_len = len(sent)

#         if sent_len <= max_seq_length:
#             zeros = list(np.zeros(max_seq_length-sent_len))
#             new = zeros+sent
#         elif sent_len > max_seq_length:
#             raise ValueError("max_seq_length too small (smaller than some sentence in the dataset)")

#         list_int_sentences_padded[i,:] = np.array(new)

#     return list_int_sentences_padded



# def create_list_encoded_words(dict, list_tokenized_sentences):
#     #dict = create_dictionary(data=data)
#     list_int_sentences = []
#     for sent in list_tokenized_sentences:
#         r = [dict[w] for w in sent]
#         list_int_sentences.append(r)

#     return list_int_sentences



# def create_list_tokenized_words(data, dict):
#     list_tokenized_sentences = []
#     max_len = 0
#     for text in data['text'].dropna():
#         list_tokenized_words = nltk.word_tokenize(text)
#         cleaned_list_tokenized_words = [word for word in list_tokenized_words
#                                         if word in dict]
#         list_tokenized_sentences.append(cleaned_list_tokenized_words)

#         if len(cleaned_list_tokenized_words)>max_len:
#             max_len = len(cleaned_list_tokenized_words)

#     logging.info(f"MAX LEN OF TOKENIZED SENTENCES: {max_len}")

#     return list_tokenized_sentences, max_len



# def create_dictionary(data):
#     words = set(nltk.corpus.words.words())
#     stop_words = stopwords.words('english')
#     list_tot_words = []
#     for text in data['text'].dropna():
#         list_tokenized_words = nltk.word_tokenize(text)
#         cleaned_list_tokenized_words = [word
#                                         for word in list_tokenized_words
#                                         if (word in words) and (word not in stop_words)
#                                         ]
#         list_tot_words += cleaned_list_tokenized_words

#     all_words = list(set(list_tot_words))
#     dictionary = {
#         word: idx+1
#         for idx, word in enumerate(all_words)
#     }

#     return dictionary


# def decode_list_int(encoded_sent, dict):
#     decoded_sent = []
#     for int in encoded_sent:
#         word = [k for k, v in dict.items() if v==int]
#         if len(word)>0:
#             decoded_sent.append(word)

#     return decoded_sent




def build_train_test_dataset(data):
    """_summary_

    Args:
        data (_type_): _description_
    """
    all_text = data['text']
    labels = data["sentiment"]
    #print(all_text_vectorized[0].shape)

    X_train, X_test, y_train, y_test = train_test_split(all_text, labels, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test



def build_train_test_count_vectorized(data, max_df=0.9, min_df=2):
    """


    Args:
        - data (pd.Dataframe): contains the all dataset
        (sentiment and text)

    Returns:
        - X_train
        - X_test
        - y_train
        - y_test
    """
    all_text = data['text']

    #Each row in matrix M contains the frequency of tokens(words) in the document D(i)

    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=None, stop_words='english')
    all_text_vectorized = count_vectorizer.fit_transform(all_text) # tokenize and build vocabulary
    #print(all_text_vectorized.shape)
    labels = data["sentiment"]
    #print(all_text_vectorized[0].shape)

    X_train, X_test, y_train, y_test = train_test_split(all_text_vectorized, labels, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test




def read_ds_twitter():
    """
    Reads the dataset (in the form of a csv)

    Args: None

    Returns:
        - train (pd.DataFrame): contains the training
        dataset (80% of the original). Has two columns
        (sentiment and text)
        - test (pd.DataFrame): contains the testing
        dataset (20% of the original). Has two
        columns (sentiment and text)
    """
    twitter_data = pd.read_csv(costants.TWITTER_CSV_PATH, sep=",", names=['_', 'sentiment', 'text'])
    twitter_data['sentiment'] = twitter_data['sentiment'].replace({'Neutral':0, 'Positive':1, 'Negative':-1})
    twitter_data = twitter_data.drop('_', axis='columns')

    #train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    return twitter_data






def read_ds():
    """
    Reads the dataset (in the form of a csv)

    Args: None

    Returns:
        - data (pd.Dataframe): financial news data
    """
    data = pd.read_csv(costants.CSV_PATH, sep=",", names=['sentiment', 'text'], encoding='latin-1')
    data['sentiment'] = data['sentiment'].replace({'neutral':0, 'positive':1, 'negative':-1})

    #train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    return data







