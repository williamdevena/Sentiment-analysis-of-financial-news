import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from src import data_processing


def grid_search_tuning_nb(data):
    """
    Performs Grid Search Hyperparameter Tuning and resturns the best accuracy.

    Args:
        data (pd.dataframe): training data

    Returns: None
    """
    max_score=0
    for max_df in  np.arange(0.1, 1.0, 0.1):
        for min_df in  np.arange(0, 10, 1):
            logging.info(f"\nMAX-DF: {max_df}")
            logging.info(f"MIN-DF: {min_df}")
            try:
                X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
                                                                                                    max_df=max_df,
                                                                                                    min_df=min_df)
                _, score = naive_bayes_classifier(X_train=X_train,
                                                X_test=X_test,
                                                y_train=y_train,
                                                y_test=y_test)

                if score>max_score:
                    max_score = score
                    best_max_df = max_df
                    best_min_df = min_df
            except:
                print("NOT POSSIBLE")


    print(f"\n\nBEST SCORE: {max_score}\nBEST MAX DF: {best_max_df}\nBEST MIN DF: {best_min_df}")
    X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
                                                                                        max_df=best_max_df,
                                                                                        min_df=best_min_df)
    y_pred, score = naive_bayes_classifier(X_train=X_train,
                                    X_test=X_test,
                                    y_train=y_train,
                                    y_test=y_test)
    build_and_save_conf_matrix(y_pred=y_pred,
                            y=y_test,
                            path="./plots/conf_matrix/nb/best_nb")






def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    """
    Trains, tests and saves the conf matrix of a Naive Bayes
    classifier.

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_

    Returns:
        - score (float): accuracy
    """
    y_pred, score = train_and_test_naive_bayes(X_train=X_train,
                                               X_test=X_test,
                                               y_train=y_train,
                                               y_test=y_test)


    logging.info(f"NAVIE BAYES ACCURACY: {score}")
    #logging.info(f"CONFUSION MATRIX SAVED IN {path_conf_matrix}")

    return y_pred, score





def train_and_test_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Trains and tests a Naive Bayes classifier.

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
    """
    model_naive = MultinomialNB().fit(X_train, y_train)
    y_pred = model_naive.predict(X_test)
    score = model_naive.score(X_test, y_test)

    return y_pred, score


def build_and_save_conf_matrix(y_pred, y, path):
    """
    Builds and saves a confusion matrix.

    Args:
        y_pred (_type_): _description_
        y_test (_type_): _description_
        path (_type_): _description_
    """
    cf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', xticklabels=["neg", "neu", "pos"], yticklabels=["neg", "neu", "pos"])
    #print(path)
    if path!=None:
        plt.savefig(path)
        plt.close()

