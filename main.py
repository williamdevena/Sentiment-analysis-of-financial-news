import logging
import os
from pprint import pprint

import pandas as pd
import transformers
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification

from models import baselines, test, transformers_pipelines
from src import data_analysis, data_loading, data_processing, pytorch_dataset
from utils import constants, metrics, setup


def main():
    # ## SETTING UP
    setup.project_setup()

    # SETTING SEED FOR REPRODUCIBILITY
    transformers.set_seed(seed=10)


    ## BASELINES ON ALLAGREE
    # data = data_processing.read_ds(agreement_percentage="sentences_allagree")
    # #print(len(data))

    # # ### SVM ON TF-IDF
    # X_train, X_test, y_train, y_test = data_processing.build_train_test_dataset(data=data)

    # avg_acc_svm, avg_precision_svm, avg_recall_svm, avg_f1_svm = baselines.svm_tf_idf(X_train=X_train,
    #                                                                  X_test=X_test,
    #                                                                  y_train=y_train,
    #                                                                  y_test=y_test,
    #                                                                  path_conf_matrix="./plots/conf_matrix/svm_tf_idf/svm_allagree")

    # ### NAIVE BAYES
    # X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
    #                                                                                      max_df=0.1,
    #                                                                                      min_df=3)
    # avg_acc_nb, avg_precision_nb, avg_recall_nb, avg_f1_nb = baselines.naive_bayes_classifier(X_train=X_train,
    #                                                                             X_test=X_test,
    #                                                                             y_train=y_train,
    #                                                                             y_test=y_test,
    #                                                                             path_conf_matrix="./plots/conf_matrix/nb/best_nb_allagree")


    # baselines_metrics = {
    #     'SVM': {'Accuracy':avg_acc_svm, 'Precision':avg_precision_svm, 'Recall':avg_recall_svm, 'F1-score':avg_f1_svm},
    #     'Naive-Bayes': {'Accuracy':avg_acc_nb, 'Precision':avg_precision_nb, 'Recall':avg_recall_nb, 'F1-score':avg_f1_nb}
    # }

    # metrics.build_and_save_radar_plot(metrics=baselines_metrics,
    #                                   path_plot=os.path.join(
    #                                       constants.PLOTS_FOLDER, "baselines_radar_plot_allagree.png"))




    # ## BASELINES ON 50AGREE
    # data = data_processing.read_ds(agreement_percentage="sentences_50agree")

    # # ### SVM ON TF-IDF
    # X_train, X_test, y_train, y_test = data_processing.build_train_test_dataset(data=data)

    # avg_acc_svm, avg_precision_svm, avg_recall_svm, avg_f1_svm = baselines.svm_tf_idf(X_train=X_train,
    #                                                                  X_test=X_test,
    #                                                                  y_train=y_train,
    #                                                                  y_test=y_test,
    #                                                                  path_conf_matrix="./plots/conf_matrix/svm_tf_idf/svm_50agree")

    # ### NAIVE BAYES
    # X_train, X_test, y_train, y_test = data_processing.build_train_test_count_vectorized(data=data,
    #                                                                                      max_df=0.1,
    #                                                                                      min_df=3)
    # avg_acc_nb, avg_precision_nb, avg_recall_nb, avg_f1_nb = baselines.naive_bayes_classifier(X_train=X_train,
    #                                                                             X_test=X_test,
    #                                                                             y_train=y_train,
    #                                                                             y_test=y_test,
    #                                                                             path_conf_matrix="./plots/conf_matrix/nb/best_nb_50agree")


    # baselines_metrics = {
    #     'SVM': {'Accuracy':avg_acc_svm, 'Precision':avg_precision_svm, 'Recall':avg_recall_svm, 'F1-score':avg_f1_svm},
    #     'Naive-Bayes': {'Accuracy':avg_acc_nb, 'Precision':avg_precision_nb, 'Recall':avg_recall_nb, 'F1-score':avg_f1_nb}
    # }

    # metrics.build_and_save_radar_plot(metrics=baselines_metrics,
    #                                   path_plot=os.path.join(
    #                                       constants.PLOTS_FOLDER, "baselines_radar_plot_50.png"))











    # ## GRID-SEARCH HYP. TUNING OF NAIVE-BAYES
    data = data_processing.read_ds(agreement_percentage="sentences_50agree")
    baselines.grid_search_tuning_nb(data=data)




    # # #### EVALUATE RoBERTa ON 50Agree
    # roberta_metrics = {}
    # tokenizer_name = "roberta-base"
    # train_ds, test_ds, val_ds = data_loading.load_train_test_val_pytorch_ds(agreement="sentences_50agree",
    #                                                             tokenizer_name=tokenizer_name)

    # model_weights = os.path.join(constants.PATH_WEIGHTS, "base_50_6_epochs")
    # y_pred, roberta_metrics["RoBERTa base (50Agree)"] = test.evaluate_model(model_path=model_weights,
    #                                             test_ds=test_ds,
    #                                             fun_compute_metrics=metrics.compute_metrics,
    #                                             path_cm=os.path.join(constants.PLOTS_FOLDER,
    #                                                                  "conf_matrix",
    #                                                                  "transformers",
    #                                                                  "RoBERTa_base_(50Agree)"),
    #                                             )

    # tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # train_ds, test_ds, val_ds = data_loading.load_train_test_val_pytorch_ds(agreement="sentences_50agree",
    #                                                            tokenizer_name=tokenizer_name)

    # model_weights = os.path.join(constants.PATH_WEIGHTS, "twitter_50_3_epochs")
    # y_pred, roberta_metrics["RoBERTa Twitter (50Agree)"] = test.evaluate_model(model_path=model_weights,
    #                                             test_ds=test_ds,
    #                                             fun_compute_metrics=metrics.compute_metrics,
    #                                             path_cm=os.path.join(constants.PLOTS_FOLDER,
    #                                                                  "conf_matrix",
    #                                                                  "transformers",
    #                                                                  "RoBERTa_Twitter_(50Agree)"),
    #                                             )

    # path_radar=os.path.join(constants.PLOTS_FOLDER, "radar_RoBERTa_50Agree.png")
    # metrics.build_and_save_radar_plot(metrics=roberta_metrics,
    #                                   path_plot=path_radar)


    # #### EVALUATE RoBERTa ON AllAgree
    # roberta_metrics = {}
    # tokenizer_name = "roberta-base"
    # train_ds, test_ds, val_ds = data_loading.load_train_test_val_pytorch_ds(agreement="sentences_allagree",
    #                                                            tokenizer_name=tokenizer_name)

    # model_weights = os.path.join(constants.PATH_WEIGHTS, "base_allagree_final")
    # y_pred, roberta_metrics["RoBERTa base (AllAgree)"] = test.evaluate_model(model_path=model_weights,
    #                                             test_ds=test_ds,
    #                                             fun_compute_metrics=metrics.compute_metrics,
    #                                             path_cm=os.path.join(constants.PLOTS_FOLDER,
    #                                                                  "conf_matrix",
    #                                                                  "transformers",
    #                                                                  "RoBERTa_base_(AllAgree)"),
    #                                             )

    # tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # train_ds, test_ds, val_ds = data_loading.load_train_test_val_pytorch_ds(agreement="sentences_allagree",
    #                                                            tokenizer_name=tokenizer_name)

    # model_weights = os.path.join(constants.PATH_WEIGHTS, "twitter_allagree_6_epochs")
    # y_pred, roberta_metrics["RoBERTa Twitter (AllAgree)"] = test.evaluate_model(model_path=model_weights,
    #                                             test_ds=test_ds,
    #                                             fun_compute_metrics=metrics.compute_metrics,
    #                                             path_cm=os.path.join(constants.PLOTS_FOLDER,
    #                                                                  "conf_matrix",
    #                                                                  "transformers",
    #                                                                  "RoBERTa_Twitter_(AllAgree)"),
    #                                             )

    # path_radar=os.path.join(constants.PLOTS_FOLDER, "radar_RoBERTa_AllAgree.png")
    # metrics.build_and_save_radar_plot(metrics=roberta_metrics,
    #                                   path_plot=path_radar)









if __name__=="__main__":
    main()
