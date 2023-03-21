import pandas as pd

from utils import costants


def read_ds():
    """
    Reads the dataset (in the form of a csv)

    Args: None

    Returns:
        - data (pd.DataFrame): contains the dataset
        has two columns (sentiment and text)
    """
    data = pd.read_csv(costants.CSV_PATH, sep=",", names=['sentiment', 'text'], encoding='latin-1')
    data['sentiment'] = data['sentiment'].replace({'neutral':0, 'positive':1, 'negative':-1})

    return data