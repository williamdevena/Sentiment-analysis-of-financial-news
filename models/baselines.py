import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

from src import data_processing
from utils import metrics

from typing import Tuple, List


def svm_tf_idf(X_train: np.ndarray,
               X_test: np.ndarray,
               y_train: np.ndarray,
               y_test: np.ndarray,
               path_conf_matrix: str) -> Tuple[float, float, float, float]:
    """
    Trains a Support Vector Machine (SVM) model using TF-IDF vectorization
    on the training data,makes predictions on the test data, and logs
    performance metrics.

    Args:
    - X_train (array): Training data feature matrix.
    - X_test (array): Test data feature matrix.
    - y_train (array): Training data target vector.
    - y_test (array): Test data target vector.
    - path_conf_matrix (str): File path to save the confusion matrix plot.

    Returns: None.
    """
    X_train_pre_processed = data_processing.pre_process_X(X=X_train)
    X_test_pre_processed = data_processing.pre_process_X(X=X_test)
    X_train_vectorized, X_test_vectorized = tf_idf_vectorize(X_train=X_train_pre_processed,
                                                             X_test=X_test_pre_processed)

    trained_svm = train_svm(X_train=X_train_vectorized,
                            y_train=y_train)

    y_pred = trained_svm.predict(X_test_vectorized)
    logging.info(f"\nSVM ON TF-IDF FEATURES")
    avg_acc, avg_precision, avg_recall, avg_f1 = metrics.log_metrics(y=y_test,
                                                                     y_pred=y_pred,
                                                                     path_conf_matrix=path_conf_matrix)

    return avg_acc, avg_precision, avg_recall, avg_f1



def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> svm.SVC:
    """
    Trains a Support Vector Machine (SVM) classifier.

    Args:
        - X_train (array-like): The feature matrix of the training data.
        - y_train (array-like): The target vector of the training data.

    Returns:
        - svm_model (svm.SVC): The trained SVM model.
    """
    svm_model = svm.SVC()
    svm_model.fit(X_train, y_train)

    return svm_model



def tf_idf_vectorize(X_train: List[str], X_test: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorizes the text data using the TF-IDF vectorizer.

    Args:
        - X_train (list): A list of training text data.
        - X_test (list): A list of test text data.

    Returns:
        - tuple: A tuple of vectorized training data and vectorized test data.
    """
    tf_idf_vectorizer = TfidfVectorizer(
        #stop_words='english'
        )
    X_train_vectorized = tf_idf_vectorizer.fit_transform(X_train)
    X_test_vectorized = tf_idf_vectorizer.transform(X_test)

    return X_train_vectorized, X_test_vectorized




def grid_search_tuning_nb(data: pd.DataFrame) -> None:
    """
    Performs Grid Search Hyperparameter Tuning and returns the best accuracy.

    Args:
        - data (pd.dataframe): training data

    Returns: None
    """
    logging.info("- PERFOMING GRID-SEARCH TUNING OF MIN_DF AND MAX_DF PARAMETERS")
    max_score=0
    list_max_df_3d = []
    list_min_df_3d = []
    list_scores_3d = []

    dict_acc = {}
    for max_df in  np.arange(0.1, 1.0, 0.1):
        list_scores = []
        list_min_df = []
        for min_df in  np.arange(0, 200, 1):  ## doesn't go more than 800
            #logging.info(f"\nMAX-DF: {max_df}")
            #logging.info(f"MIN-DF: {min_df}")
            try:
                X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
                                                                                                    max_df=0.1,
                                                                                                    min_df=min_df)
                score, _, _, _ = naive_bayes_classifier(X_train=X_train,
                                                X_test=X_test,
                                                y_train=y_train,
                                                y_test=y_test,
                                                path_conf_matrix=None,
                                                log_metrics=False)

                if score>max_score:
                    max_score = score
                    best_max_df = max_df
                    best_min_df = min_df

                #logging.info(f"SCORE: {score}")
                list_scores.append(score)
                list_min_df.append(min_df)
                list_max_df_3d.append(max_df)
                list_min_df_3d.append(min_df)
                list_scores_3d.append(score)
            except:
                print("Error")

        dict_acc[max_df] = (list_scores, list_min_df)

    logging.info(f"\n\nBEST SCORE: {max_score}\nBEST MAX DF: {best_max_df}\nBEST MIN DF: {best_min_df}")

    for idx, (max_df, (list_scores, list_min_df)) in enumerate(dict_acc.items()):
        plt.plot(list_min_df, list_scores, label=f"{max_df:.1f} Max DF")

    plt.legend()
    plt.xlabel("Min DF")
    plt.ylabel("Accuracy")
    plt.title('Naive-Bayes Grid-Search 2D')
    plt.savefig("./plots/nb_hyp_tuning/plot_nb_grid_search_2d")
    plt.close()

    fig = plt.figure()
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    p = ax.scatter(np.array(list_max_df_3d),
               np.array(list_min_df_3d),
               np.array(list_scores_3d),
               c=np.array(list_scores_3d))

    ax.set_xlabel('Max DF')
    ax.set_ylabel('Min DF')
    ax.set_zlabel('Accuracy')
    ax.set_title('Naive-Bayes Grid-Search 3D')
    cbar = fig.colorbar(p, location='left')
    cbar.set_label('Acc', rotation=270)
    plt.savefig("./plots/nb_hyp_tuning/plot_nb_grid_search_3d")
    plt.close()






def naive_bayes_classifier(X_train: np.ndarray,
                           X_test: np.ndarray,
                           y_train: np.ndarray,
                           y_test: np.ndarray,
                           path_conf_matrix: str,
                           log_metrics: bool = True) -> Tuple[float, float, float, float]:
    """
    Trains and tests a Naive Bayes classifier on the given data.

    Args:
        - X_train (array): Training data feature matrix.
        - X_test (array): Test data feature matrix.
        - y_train (array): Training data target vector.
        - y_test (array): Test data target vector.
        - path_conf_matrix (str): File path to save the confusion matrix plot.
        - log_metrics (bool): if False the metrics are not logged and
        the conf. matrix is not saved.

    Returns:
        - score (float): Average accuracy.
    """
    # X_train_pre_processed = data_processing.pre_process_X(X=X_train)
    # X_test_pre_processed = data_processing.pre_process_X(X=X_test)
    y_pred, score = train_and_predict_naive_bayes(X_train=X_train,
                                                X_test=X_test,
                                                y_train=y_train,
                                                y_test=y_test)
    if log_metrics:
        logging.info(f"\nNAIVE-BAYES CLASSIFIER")
    avg_acc, avg_precision, avg_recall, avg_f1 = metrics.log_metrics(y=y_test,
                                                                        y_pred=y_pred,
                                                                        path_conf_matrix=path_conf_matrix,
                                                                        log=False)

    return avg_acc, avg_precision, avg_recall, avg_f1




def train_and_predict_naive_bayes(X_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_train: np.ndarray,
                                  y_test: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Trains a Naive Bayes classifier and makes predictions.

    Args:
        - X_train (array): Training data feature matrix.
        - X_test (array): Test data feature matrix.
        - y_train (array): Training data target vector.
        - y_test (array): Testing data target vector.

    Returns:
        - y_pred (array): Predicted target vector for test data.
        - score (float): Average accuracy.
    """
    model_naive = MultinomialNB().fit(X_train, y_train)
    y_pred = model_naive.predict(X_test)
    score = model_naive.score(X_test, y_test)

    return y_pred, score




