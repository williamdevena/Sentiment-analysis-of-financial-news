import logging
import os

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from models import transformers_pipelines
from src import baselines, data_analysis, data_processing, pytorch_dataset
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


    ## STATISTICS DATASET
    # data = data_processing.read_ds()
    # tot_words, counts_words = data_analysis.calculate_stats_words(data)
    ####print(len(tot_words), counts_words[:10])

    # ## STATISTICS TWITTER DATA
    # twitter_data = data_processing.read_ds_twitter()
    # tot_words, counts_words = data_analysis.calculate_stats_words(twitter_data)
    # ###print(len(tot_words), counts_words[:10])


    # ## COMPARISON DATASETS
    # financial_data = data_processing.read_ds()
    # twitter_data = data_processing.read_ds_twitter()
    # exclusive_words_financial, exclusive_words_twitter = data_analysis.compare_datasets(financial_data, twitter_data)
    # print(exclusive_words_financial, exclusive_words_twitter)
    # print(len(exclusive_words_financial), len(exclusive_words_twitter))


    # # ### BASELINESS
    # data = data_processing.read_ds()

    # # ### SVM ON TF-IDF
    # X_train, X_test, y_train, y_test = data_processing.build_train_test_dataset(data=data)
    # baselines.svm_tf_idf(X_train=X_train,
    #                      X_test=X_test,
    #                      y_train=y_train,
    #                      y_test=y_test,
    #                      path_conf_matrix="./plots/conf_matrix/svm_tf_idf/svm")

    # # ### NAIVE BAYES
    # X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
    #                                                                                      max_df=0.1,
    #                                                                                      min_df=3)
    # baselines.naive_bayes_classifier(X_train=X_train,
    #                                 X_test=X_test,
    #                                 y_train=y_train,
    #                                 y_test=y_test,
    #                                 path_conf_matrix="./plots/conf_matrix/nb/best_nb_2")






    #### GRID-SEARCH HYP. TUNING OF NAIVE-BAYES
    # data = data_processing.read_ds()
    # baselines.grid_search_tuning_nb(data=data)



    # ### CODE TO SAVE DIVIDED TRAIN AND TEST (FOR PYTORCH DS)
    # print(X_train.index, y_train.index)
    # #print(pd.merge(X_train, y_train, left_index=True, right_index=True))
    # train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    # test = pd.merge(X_test, y_test, left_index=True, right_index=True)

    # train.to_csv("./train.csv")
    # test.to_csv("./test.csv")

    # print(train.shape, test.shape)





    ### TEST HUGGING FACE PIPELINES
    # data = data_processing.read_ds()
    # X_train, X_test, y_train, y_test = data_processing.build_train_test_dataset(data=data)


    # ## TWITTER ROBERTA
    # transformers_pipelines.test_hugging_face_pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    #                                                 X_test=X_test,
    #                                                 y_test=y_test,
    #                                                 path_conf_matrix="./plots/conf_matrix/transformers/twitter-roberta"
    #                                                 )

    # ## FIN-BERT
    # transformers_pipelines.test_hugging_face_pipeline(model="ahmedrachid/FinancialBERT-Sentiment-Analysis",
    #                                                 X_test=X_test,
    #                                                 y_test=y_test,
    #                                                 path_conf_matrix="./plots/conf_matrix/transformers/finBERT"
    #                                                 )




    ### PYTORCH DATASET
    ds = pytorch_dataset.FinancialNewsDataset(path_csv=costants.FINANCIAL_NEWS_TRAIN_DATA)
    print(len(ds))
    print(ds[4])











if __name__=="__main__":
    main()
