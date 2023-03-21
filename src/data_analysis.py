import nltk
import numpy as np


def ds_statistics(ds):
    """
    Returns some statistics on the dataset:

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


def calculate_stats_length_text(ds):
    """
    Calculates statistics on the length of the text
    in the dataset.

    Args:
        - ds (pd.DataFrame): the dataset, has two
        columns (sentiment and text)

    Returns:
        - dict_stats (Dict): contains the statistics
    """
    text_col = np.array(ds['text'])
    len_text = [len(text) for text in text_col]
    max_len = len_text.max()
    min_len = len_text.min()
    mean_len = len_text.mean()

    return {
        'max_len_text': max_len,
        'min_len_text': min_len,
        'mean_len_text': mean_len
    }



def calculate_stats_words(ds):
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
    for text in ds['text']:
        list_tot_words += nltk.word_tokenize(text)

    counts_words = np.unique(list_tot_words, return_counts=True)
    counts_words = sorted(list(zip(*counts_words)), key=lambda x: x[1], reverse=True)

    return list_tot_words, counts_words




