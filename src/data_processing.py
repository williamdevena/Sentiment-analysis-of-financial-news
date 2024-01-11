import logging
from copy import deepcopy

import nltk
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy

from utils import constants

from typing import List, Dict, Tuple

def pre_process_X(X: List[str]) -> np.ndarray:
    """
    Pre-processes the input text data X by lemmatizing each word and joining them back into a single string.

    Args:
        X (list): A list of strings, each string representing a text document.

    Returns:
        numpy.ndarray: An array of pre-processed strings of shape (n_samples,).
    """
    #stemmer = nltk.stem.PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    X_pre_processed = []
    for text in X:
        list_tokens = nltk.word_tokenize(text)
        list_stemmed_tokens = [lemmatizer.lemmatize(token) for token in list_tokens]
        new_text = " ".join(list_stemmed_tokens)
        X_pre_processed.append(new_text)

    return np.array(X_pre_processed)





def build_dict(filename: str) -> Dict[str, np.ndarray]:
    """
    Builds a dictionary of word vectors from a file.

    Args:
        filename (str): The path to the file containing the word
        vectors.

    Returns:
        dict: A dictionary containing word vectors as values and
        corresponding words as keys.
    """
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


def message_to_token_list(sent: str, words_dict: Dict[str, np.ndarray]) -> List[str]:
    """
    Converts a text message into a list of lemmatized tokens,
    filtered by the words present in the word dictionary.

    Args:
        sent (str): A text message to be tokenized.
        words_dict (dict): A dictionary containing word vectors
        as values and corresponding words as keys.

    Returns:
        list: A list of lemmatized tokens from the message.
    """
    lemmatizer = WordNetLemmatizer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(sent)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words_dict]

    return useful_tokens


def message_to_word_vectors(sent: str, words_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Converts a text message into a list of corresponding word vectors,
    filtered by the words present in the word dictionary.

    Args:
        sent (str): A text message to be vectorized.
        words_dict (dict): A dictionary containing word vectors as
        values and corresponding words as keys.

    Returns:
        numpy.ndarray: An array of corresponding word vectors of
        shape (n_tokens, n_features).
    """
    processed_list_of_tokens = message_to_token_list(sent=sent,
                                                     words_dict=words_dict)
    vectors = []
    for token in processed_list_of_tokens:
        if token not in words_dict:
            continue
        token_vector = words_dict[token]
        vectors.append(token_vector)

    return np.array(vectors, dtype=float)



def vectorize_ds(path_csv: str, words_dict: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Loads a dataset from a CSV file and vectorizes the text data
    using the specified word dictionary.

    Args:
        path_csv (str): The path to the CSV file containing the dataset.
        words_dict (dict): A dictionary containing word vectors as
        values and corresponding words as keys.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the
        vectorized text data and corresponding sentiment labels.
    """
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




def pad_list_embeddings(list_embeddings: List[np.ndarray], desired_sent_length: int) -> np.ndarray:
    """
    Pads a list of sentence embeddings with zeros to a desired sentence length.

    Args:
        list_embeddings (List[np.ndarray]): A list of numpy arrays representing sentence embeddings.
        desired_sent_length (int): The desired length for each sentence.

    Returns:
        np.ndarray: A 3D numpy array of shape (num_sentences, desired_sent_length, embedding_size) representing the padded sentence embeddings.
    """
    list_embeddings_padded = deepcopy(list_embeddings)

    for i, sent in enumerate(list_embeddings):
        x_seq_len = sent.shape[0]
        sent_length_difference = desired_sent_length - x_seq_len

        pad = np.zeros(shape=(sent_length_difference, 50))

        list_embeddings_padded[i] = np.concatenate([sent, pad])

    return np.array(list_embeddings_padded).astype(float)




def build_train_test_dataset(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the data into training and test sets and returns them along with their corresponding labels.

    Args:
        data (pandas.DataFrame): The data to be split.

    Returns:
        tuple: A tuple of numpy arrays representing the training and test sets and their corresponding labels.
    """
    all_text = data['text']
    labels = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(all_text, labels, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test



def build_train_test_count_vectorized(data: pd.DataFrame,
                                      max_df: float = 0.9,
                                      min_df: int = 2) -> Tuple[scipy.sparse.csr.csr_matrix, scipy.sparse.csr.csr_matrix, pd.Series, pd.Series]:
    """
    Vectorizes the text data using count vectorization and splits it into
    training and test sets. Returns the vectorized training and test sets
    and their corresponding labels.

    Args:
        data (pandas.DataFrame): The data to be vectorized and split.
        max_df (float, optional): When building the vocabulary, ignore
            terms that have a document frequency strictly higher than
            the given threshold (corpus-specific stop words). Defaults to 0.9.
        min_df (int, optional): When building the vocabulary, ignore terms
            that have a document frequency strictly lower than the given
            threshold. This value is also called cut-off in the literature.
            Defaults to 2.

    Returns:
        tuple: A tuple of numpy arrays representing the vectorized training
        and test sets and their corresponding labels.
    """
    all_text = data['text']

    #Each row in matrix M contains the frequency of tokens(words) in the document D(i)
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=None, stop_words='english')
    all_text_vectorized = count_vectorizer.fit_transform(all_text) # tokenize and build vocabulary
    labels = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(all_text_vectorized, labels, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test




def read_ds_twitter() -> pd.DataFrame:
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
    twitter_data = pd.read_csv(constants.TWITTER_CSV_PATH, sep=",", names=['_', 'sentiment', 'text'])
    twitter_data['sentiment'] = twitter_data['sentiment'].replace({'Neutral':0, 'Positive':1, 'Negative':-1})
    twitter_data = twitter_data.drop('_', axis='columns')

    return twitter_data






def read_ds(agreement_percentage: float) -> pd.DataFrame:
    """
    Reads the dataset (in the form of a csv)

    Args: None

    Returns:
        - data (pd.Dataframe): financial news data
    """
    # Load the dataset from the Hugging Face hub
    ds = load_dataset("financial_phrasebank", agreement_percentage)
    data = ds['train'].to_pandas()
    data = data.rename(columns={'sentence': 'text', 'label': 'sentiment'})


    return data








