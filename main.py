import pandas as pd

from src import data_analysis, data_processing
from utils import costants


def main():
    data = data_processing.read_ds()
    print(data)
    tot_words, counts_words = data_analysis.calculate_stats_words(data)
    print(len(tot_words), counts_words[:10])

    ## RED DATA



if __name__=="__main__":
    main()