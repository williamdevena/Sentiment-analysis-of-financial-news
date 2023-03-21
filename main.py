import pandas as pd

from utils import costants


def main():
    data = pd.read_csv(costants.CSV_PATH, sep="|", encoding='latin-1')
    print(data)

    ## RED DATA



if __name__=="__main__":
    main()