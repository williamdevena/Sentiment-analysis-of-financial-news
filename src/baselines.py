import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB


def naive_bayes_classifier(X_train, X_test, y_train, y_test, path_conf_matrix):
    """
    Trains, tests and saves the conf matrix of a Naive Bayes
    classifier.

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
    """
    y_pred, score = train_and_test_naive_bayes(X_train=X_train,
                                               X_test=X_test,
                                               y_train=y_train,
                                               y_test=y_test)
    build_and_save_conf_matrix(y_pred=y_pred,
                               y=y_test,
                               path=path_conf_matrix)

    logging.info(f"NAVIE BAYES ACCURACY: {score}")
    logging.info(f"CONFUSIO MATRIX SAVED IN {path_conf_matrix}")





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
    plt.savefig(path)

