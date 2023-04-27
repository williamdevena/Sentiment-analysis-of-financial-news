import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils import costants


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



def build_train_test_count_vectorized(data, max_df, min_df):
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





def read_ds():
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
    data = pd.read_csv(costants.CSV_PATH, sep=",", names=['sentiment', 'text'], encoding='latin-1')
    data['sentiment'] = data['sentiment'].replace({'neutral':0, 'positive':1, 'negative':-1})

    #train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    return data



