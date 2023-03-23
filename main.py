import logging

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src import baselines, data_analysis, data_processing
from utils import costants


def main():

    ## LOGGING SETUP
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("project_logs/assignment.log"),
            logging.StreamHandler()
        ]
    )


    #train, test = data_processing.read_ds()
    #print(data)

    ## STATISTICS DATASET
    # tot_words, counts_words = data_analysis.calculate_stats_words(data)
    # print(len(tot_words), counts_words[:10])



    # sia = SentimentIntensityAnalyzer()
    # #res = {}
    # list_dict = []
    # pred = []
    # for i, text in enumerate(data['text']):
    #     #print(i, text)
    #     #text = row['text']
    #     #myid = row['Id']
    #     res = sia.polarity_scores(text)
    #     res.pop('compound')
    #     list_dict.append(res)
    #     y_pred = max(res, key=lambda x: res[x])
    #     #if y_pred==
    #     pred.append(y_pred)

    # print(pred, list_dict)



    data = data_processing.read_ds()
    X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    baselines.naive_bayes_classifier(X_train=X_train,
                                    X_test=X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                    path_conf_matrix="./nb_conf_matrix")

    #print(nb_score)




if __name__=="__main__":
    main()