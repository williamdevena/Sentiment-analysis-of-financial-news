import os

PROJECT_PATH = os.path.abspath(".")
DATASET_PATH = os.path.join(PROJECT_PATH, "../Data")
CSV_PATH = os.path.join(DATASET_PATH, "all-data.csv")
TWITTER_CSV_PATH = os.path.join(DATASET_PATH, "twitter_training.csv")
FINANCIAL_NEWS_TRAIN_DATA = os.path.join(DATASET_PATH, "train.csv")
FINANCIAL_NEWS_TEST_DATA = os.path.join(DATASET_PATH, "test.csv")

PLOTS_FOLDER = os.path.join(PROJECT_PATH, "plots")
STATS_PLOT_FOLDER = os.path.join(PLOTS_FOLDER, "ds_stats")


