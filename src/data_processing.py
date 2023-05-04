import logging

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils import costants


def pad_sentences(list_int_sentences, max_seq_length):
    list_int_sentences_padded = np.zeros((len(list_int_sentences), max_seq_length), dtype = int)

    for i, sent in enumerate(list_int_sentences):
        sent_len = len(sent)

        if sent_len <= max_seq_length:
            zeros = list(np.zeros(max_seq_length-sent_len))
            new = zeros+sent
        elif sent_len > max_seq_length:
            raise ValueError("max_seq_length too small (smaller than some sentence in the dataset)")

        list_int_sentences_padded[i,:] = np.array(new)

    return list_int_sentences_padded



def create_list_encoded_words(dict, list_tokenized_sentences):
    #dict = create_dictionary(data=data)
    list_int_sentences = []
    for sent in list_tokenized_sentences:
        r = [dict[w] for w in sent]
        list_int_sentences.append(r)

    return list_int_sentences



def create_list_tokenized_words(data, dict):
    list_tokenized_sentences = []
    max_len = 0
    for text in data['text'].dropna():
        list_tokenized_words = nltk.word_tokenize(text)
        cleaned_list_tokenized_words = [word for word in list_tokenized_words
                                        if word in dict]
        list_tokenized_sentences.append(cleaned_list_tokenized_words)

        if len(cleaned_list_tokenized_words)>max_len:
            max_len = len(cleaned_list_tokenized_words)

    logging.info(f"MAX LEN OF TOKENIZED SENTENCES: {max_len}")

    return list_tokenized_sentences, max_len



def create_dictionary(data):
    words = set(nltk.corpus.words.words())
    stop_words = stopwords.words('english')
    list_tot_words = []
    for text in data['text'].dropna():
        list_tokenized_words = nltk.word_tokenize(text)
        cleaned_list_tokenized_words = [word
                                        for word in list_tokenized_words
                                        if (word in words) and (word not in stop_words)
                                        ]
        list_tot_words += cleaned_list_tokenized_words

    all_words = list(set(list_tot_words))
    dictionary = {
        word: idx+1
        for idx, word in enumerate(all_words)
    }

    return dictionary


def decode_list_int(encoded_sent, dict):
    decoded_sent = []
    for int in encoded_sent:
        word = [k for k, v in dict.items() if v==int]
        if len(word)>0:
            decoded_sent.append(word)

    return decoded_sent




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







