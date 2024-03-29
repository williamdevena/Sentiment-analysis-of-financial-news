import logging
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

from utils import constants
from typing import List, Tuple


def stats_embeddings(list_embeddings: list[np.ndarray], path_plot: str) -> None:
    """
    Generates descriptive statistics of a list of embeddings.

    Args:
        list_embeddings (list): A list of numpy arrays representing sentence embeddings.
        path_plot (str): The file path to save the histogram plot of sentence lengths.

    Returns:
        None

    """
    list_lengths_sentences = [sent.shape[0] for sent in list_embeddings]
    plt.hist(list_lengths_sentences)
    plt.savefig(path_plot)
    plt.close()
    series_embeddings = pd.Series(list_lengths_sentences)
    logging.info("- STATISTICS OF VECTORIZED DATASET:")
    logging.info(series_embeddings.describe())






def ds_statistics(ds: pd.DataFrame) -> Tuple[dict, pd.Series, pd.Series]:
    """
    Returns some statistics on the dataset: length of sentences,
    distribution of lables, statistics on the signle words.

    Args:
        - ds (pd.DataFrame): the dataset, has two
        columns (sentiment and text)

    Returns:
        - stats_length_text (Dict): contain max, min and mean
        of the length of the text in the ds
        - stats_labels (pd.Series)
        - stats_single_words
    """

    ## calculates statistics on length of text
    stats_length_text = calculate_stats_length_text(ds)

    ## calculates statistics on labels
    stats_labels = ds['sentiment'].value_counts()

    ## calculates statistics on single words
    stats_single_words = calculate_stats_words(ds)

    return stats_length_text, stats_labels, stats_single_words


def calculate_stats_length_text(ds: pd.DataFrame, path_hist_plot: str) -> dict:
    """
    Calculates statistics on the length of the text
    in the dataset.

    Args:
        - ds (pd.DataFrame): the dataset, has two
        columns (sentiment and text)

    Returns:
        - dict_stats (Dict): contains the statistics
    """
    text_col = np.array(ds['text'].dropna())
    len_text = np.array([len(text) for text in text_col])
    plt.hist(len_text)
    plt.xlabel("Length")
    plt.ylabel("# Sentences")
    plt.savefig(path_hist_plot)
    plt.close()
    max_len = len_text.max()
    min_len = len_text.min()
    mean_len = len_text.mean()

    logging.info(f"MEAN LENGTH OF SENTENCE: {mean_len}")
    logging.info(f"MAX LENGTH OF SENTENCE: {max_len}")
    logging.info(f"MIN LENGTH OF SENTENCE: {min_len}")

    return {
        'max_len_text': max_len,
        'min_len_text': min_len,
        'mean_len_text': mean_len
    }



def calculate_stats_words(ds: pd.DataFrame) -> Tuple[List[str], dict, int]:
    """
    Calculates statistics on the single words
    in the dataset.

    Args:
        - ds (pd.DataFrame): the dataset, has two
        columns (sentiment and text)

    Returns:
        - list_tot_words (List): contains all
        the world in the ds (with repetitions)
    """
    list_tot_words = []
    for text in ds['text'].dropna():
        list_tot_words += nltk.word_tokenize(text)

    all_words = list(set(list_tot_words))
    counts_words = np.unique(list_tot_words, return_counts=True)

    counts_words = {
        word: count
        for word, count in zip(counts_words[0], counts_words[1])
    }

    logging.info(f"TOTAL NUMBER OF DIFFERENT WORDS: {len(all_words)}")

    return all_words, counts_words, len(list_tot_words)




def compare_datasets(ds_1: pd.DataFrame, ds_2: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Compares two datasets based on the frequency of words that appear in them.

    Args:
        - ds_1 (pd.DataFrame): The first dataset to compare.
        - ds_2 (pd.DataFrame): The second dataset to compare.

    Returns:
        A tuple containing two lists:
            - The words that appear exclusively in ds_1.
            - The words that appear exclusively in ds_2.
    """
    words = set(nltk.corpus.words.words())
    logging.info("FINANCIAL DATA")
    tot_diff_words_1, counts_words_1, tot_num_words_1 = calculate_stats_words(ds_1)
    _ = calculate_stats_length_text(ds=ds_1,
                                    path_hist_plot=os.path.join(constants.STATS_PLOT_FOLDER, "financial_news"))
    logging.info("\nTWITTER DATA")
    tot_diff_words_2, counts_words_2, tot_num_words_2 = calculate_stats_words(ds_2)
    _ = calculate_stats_length_text(ds=ds_2,
                                    path_hist_plot=os.path.join(constants.STATS_PLOT_FOLDER, "twitter"))

    threshold = 0.00008
    exclusive_words_1 = [word for word in tot_diff_words_1 if ((word not in tot_diff_words_2) and
                                                          (word in words) and
                                                          (counts_words_1[word]/tot_num_words_1>threshold)
                                                          )]

    exclusive_words_2 = [word for word in tot_diff_words_2 if ((word not in tot_diff_words_1) and
                                                               (word in words) and
                                                               (counts_words_2[word]/tot_num_words_2>threshold)
                                                               )]

    return exclusive_words_1, exclusive_words_2





