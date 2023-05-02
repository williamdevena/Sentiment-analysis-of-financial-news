import logging

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src import data_processing
from utils import metrics


def svm_tf_idf(X_train, X_test, y_train, y_test, path_conf_matrix):
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
    X_train_vectorized, X_test_vectorized = tf_idf_vectorize(X_train=X_train,
                                                             X_test=X_test)

    trained_svm = train_svm(X_train=X_train_vectorized,
                            y_train=y_train)

    y_pred = trained_svm.predict(X_test_vectorized)
    logging.info(f"\nSVM ON TF-IDF FEATURES")
    metrics.log_metrics(y=y_test, y_pred=y_pred, path_conf_matrix=path_conf_matrix)



def train_svm(X_train, y_train):
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



def tf_idf_vectorize(X_train, X_test):
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




def grid_search_tuning_nb(data):
    """
    Performs Grid Search Hyperparameter Tuning and returns the best accuracy.

    Args:
        - data (pd.dataframe): training data

    Returns: None
    """
    max_score=0
    list_max_df = []
    list_min_df = []
    list_scores = []
    for max_df in  np.arange(0.1, 1.0, 0.01):
    #for max_df in  np.arange(4, 3000, 1):
        for min_df in  np.arange(0, 800, 1):  ## doesn't go more than 800
            logging.info(f"\nMAX-DF: {max_df}")
            logging.info(f"MIN-DF: {min_df}")
            # try:
            X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
                                                                                                max_df=max_df,
                                                                                                min_df=min_df)
            score = naive_bayes_classifier(X_train=X_train,
                                            X_test=X_test,
                                            y_train=y_train,
                                            y_test=y_test,
                                            path_conf_matrix=None,
                                            log_metrics=False)
            logging.info(f"SCORE: {score}")
            list_scores.append(score)
            list_max_df.append(max_df)
            list_min_df.append(min_df)

            if score>max_score:
                max_score = score
                best_max_df = max_df
                best_min_df = min_df
            # except:
            #     print("NOT POSSIBLE")


    print(f"\n\nBEST SCORE: {max_score}\nBEST MAX DF: {best_max_df}\nBEST MIN DF: {best_min_df}")
    X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
                                                                                        max_df=best_max_df,
                                                                                        min_df=best_min_df)

    # plt.plot(list_max_df, list_scores)
    # plt.xlabel("Max DF")
    # plt.ylabel("Accuracy")
    # plt.savefig("./2d_nb_max_df")
    # fig = px.scatter_3d(x=list_max_df, y=list_min_df, z=list_scores)
    # fig.show()

    # fig = plt.figure()
    # # syntax for 3-D projection
    # ax = plt.axes(projection ='3d')
    # # ax.plot_trisurf(np.array(list_max_df), np.array(list_min_df), np.array(list_scores),
    # #                cmap='viridis', edgecolor='green')
    # # ax.scatter(np.array(list_max_df),
    # #            np.array(list_min_df),
    # #            np.array(list_scores),
    # #            c=np.array(list_scores))
    # ax.plot3D(list_max_df, list_min_df, list_scores, 'green')

    # ax.set_xlabel('Max DF')
    # ax.set_ylabel('Min DF')
    # ax.set_zlabel('Accuracy')
    # ax.set_title('Naive-Bayes Grid-Search Hyperparameters')
    # plt.savefig("./3d_nb2")






def naive_bayes_classifier(X_train, X_test, y_train, y_test, path_conf_matrix, log_metrics=True):
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
    y_pred, score = train_and_predict_naive_bayes(X_train=X_train,
                                                X_test=X_test,
                                                y_train=y_train,
                                                y_test=y_test)
    if log_metrics:
        logging.info(f"\nNAIVE-BAYES CLASSIFIER")
        metrics.log_metrics(y=y_test, y_pred=y_pred, path_conf_matrix=path_conf_matrix)

    return score




def train_and_predict_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Trains and tests a Naive Bayes classifier on the given data.

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




