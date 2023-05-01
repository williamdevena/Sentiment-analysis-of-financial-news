import logging

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src import (baselines, data_analysis, data_processing,
                 hugging_face_pipelines)
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



    # STATISTICS DATASET
    # data = data_processing.read_ds()
    # tot_words, counts_words = data_analysis.calculate_stats_words(data)
    # print(len(tot_words), counts_words[:10])

    # # STATISTICS TWITTER DATA
    # twitter_data = data_processing.read_ds_twitter()
    # tot_words, counts_words = data_analysis.calculate_stats_words(twitter_data)
    # print(len(tot_words), counts_words[:10])


    ## COMPARISON DATASETS
    # financial_data = data_processing.read_ds()
    # twitter_data = data_processing.read_ds_twitter()
    # exclusive_words_financial, exclusive_words_twitter = data_analysis.compare_datasets(financial_data, twitter_data)
    # print(exclusive_words_financial)
    # print(len(exclusive_words_financial), len(exclusive_words_twitter))



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





    #### GRID-SEARCH HYP. TUNING OF NAIVE-BAYES
    # data = data_processing.read_ds()
    # baselines.grid_search_tuning_nb(data=data)





    ### TEST HUGGING FACE PIPELINES
    # data = data_processing.read_ds()
    # X_train, X_test, y_train, y_test = data_processing.build_train_test_dataset(data=data)
    # acc = hugging_face_pipelines.test_hugging_face_pipeline(#model="ahmedrachid/FinancialBERT-Sentiment-Analysis",
    #                                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    #                                                         X_test=X_test[:10],
    #                                                         y_test=y_test[:10],
    #                                                         labelling_function=hugging_face_pipelines.labelling_function_twitter_roberta)
    # print(acc)






    # ### BASELINESS
    data = data_processing.read_ds()

    # ### SVM ON TF-IDF
    X_train, X_test, y_train, y_test = data_processing.build_train_test_dataset(data=data)
    baselines.svm_tf_idf(X_train=X_train,
                         X_test=X_test,
                         y_train=y_train,
                         y_test=y_test,
                         path_conf_matrix="./plots/conf_matrix/svm_tf_idf/svm")

    # ### NAIVE BAYES
    X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
                                                                                         max_df=0.1,
                                                                                         min_df=3)
    baselines.naive_bayes_classifier(X_train=X_train,
                                    X_test=X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                    path_conf_matrix="./plots/conf_matrix/nb/best_nb_2")








if __name__=="__main__":
    main()
