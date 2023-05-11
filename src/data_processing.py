import logging
from copy import deepcopy

import nltk
import numpy as np
import pandas as pd
import torch
import torchtext
from datasets import DatasetDict, load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils import costants


def pre_process_X(X):
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





def build_dict(filename):
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


def message_to_token_list(sent, words_dict):
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


def message_to_word_vectors(sent, words_dict):
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



def vectorize_ds(path_csv, words_dict):
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




def pad_list_embeddings(list_embeddings, desired_sent_length):
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
    """
    Splits the data into training and test sets and returns them along with their corresponding labels.

    Args:
        data (pandas.DataFrame): The data to be split.

    Returns:
        tuple: A tuple of numpy arrays representing the training and test sets and their corresponding labels.
    """
    all_text = data['text']
    labels = data["sentiment"]
    #print(all_text_vectorized[0].shape)

    X_train, X_test, y_train, y_test = train_test_split(all_text, labels, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test



def build_train_test_count_vectorized(data, max_df=0.9, min_df=2):
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
    #all_text = pre_process_X(X=all_text)

    #Each row in matrix M contains the frequency of tokens(words) in the document D(i)
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=None, stop_words='english')
    all_text_vectorized = count_vectorizer.fit_transform(all_text) # tokenize and build vocabulary
    labels = data["sentiment"]

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






def read_ds(agreement_percentage):
    """
    Reads the dataset (in the form of a csv)

    Args: None

    Returns:
        - data (pd.Dataframe): financial news data
    """
    # data = pd.read_csv(costants.CSV_PATH, sep=",", names=['sentiment', 'text'], encoding='latin-1')
    # data['sentiment'] = data['sentiment'].replace({'neutral':0, 'positive':1, 'negative':-1})

    # Load the dataset from the Hugging Face hub
    ds = load_dataset("financial_phrasebank", agreement_percentage)
    data = ds['train'].to_pandas()
    data = data.rename(columns={'sentence': 'text', 'label': 'sentiment'})

    #train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    return data



# def read_ds():
#     """
#     Reads the dataset (in the form of a csv)

#     Args: None

#     Returns:
#         - data (pd.Dataframe): financial news data
#     """
#     data = pd.read_csv(costants.CSV_PATH, sep=",", names=['sentiment', 'text'], encoding='latin-1')
#     data['sentiment'] = data['sentiment'].replace({'neutral':0, 'positive':1, 'negative':-1})

#     #train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
#     print(len(data))

#     return data







