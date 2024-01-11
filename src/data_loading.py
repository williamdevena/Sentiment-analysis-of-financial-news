from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
import transformers

from src.pytorch_dataset import FinancialNewsDataset
from typing import Tuple
import pandas as pd
from typing import List


def load_train_test_val_pytorch_ds(agreement: str,
                                   tokenizer_name: str) -> Tuple[FinancialNewsDataset, FinancialNewsDataset, FinancialNewsDataset]:
    """
    Load the Financial Phrasebank dataset and split it into train,
    test, and validation sets for PyTorch.

    Args:
        - agreement (str): The level of agreement among annotators. Can
            be "sentences_allagree" or "sentences_50agree".
        - tokenizer_name (str): The name of the pre-trained tokenizer to use.

    Returns:
        - tuple: A tuple of PyTorch datasets for the train, test,
            and validation sets.

    """
    ds = load_dataset("financial_phrasebank", agreement)
    ds_dict = split_ds(ds=ds)
    df_train, df_test, df_val = from_dsdict_to_dataframe(
        ds_dict=ds_dict)
    train_texts, train_labels = list(df_train['sentence']), list(df_train['label'])
    test_texts, test_labels = list(df_test['sentence']), list(df_test['label'])
    val_texts, val_labels = list(df_val['sentence']), list(df_val['label'])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_ds = build_pytorch_dataset(texts=train_texts,
                                     labels=train_labels,
                                     tokenizer=tokenizer)
    test_ds = build_pytorch_dataset(texts=test_texts,
                                     labels=test_labels,
                                     tokenizer=tokenizer)
    val_ds = build_pytorch_dataset(texts=val_texts,
                                     labels=val_labels,
                                     tokenizer=tokenizer)

    return train_ds, test_ds, val_ds



def split_ds(ds) -> DatasetDict:
    """
    Split a Hugging Face dataset into train, test, and validation sets.

    Args:
        - ds (Dataset): A Hugging Face dataset object.

    Returns:
        - DatasetDict: A dictionary containing the train, test, and
            validation sets.
    """
    train_validation_test = ds['train'].train_test_split(shuffle=True, seed=123, test_size=0.1)
    train_validation_test
    ds_dict = DatasetDict({'train': train_validation_test['train'],
                            'test': train_validation_test['test']})
    valid=ds_dict['train'].train_test_split(test_size=0.11, seed=123)
    ds_dict = DatasetDict({'train': valid['train'],
                            'validation': valid['test'],
                            'test': train_validation_test['test']})
    return ds_dict



def from_dsdict_to_dataframe(ds_dict: DatasetDict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert a DatasetDict object to three Pandas dataframes for
    the train, test, and validation sets.

    Args:
        - ds_dict (DatasetDict): A dictionary containing the train,
        test, and validation sets.

    Returns:
        - tuple: A tuple of three Pandas dataframes for the train,
            test, and validation sets.
    """
    df_train = ds_dict['train'].to_pandas()
    df_test = ds_dict['test'].to_pandas()
    df_val = ds_dict['validation'].to_pandas()

    return df_train, df_test, df_val


def build_pytorch_dataset(texts: List[str],
                          labels: List[str],
                          tokenizer: transformers.PreTrainedTokenizer) -> FinancialNewsDataset:
    """
    Builds a PyTorch dataset from a list of texts and their corresponding labels, using the specified tokenizer
    to tokenize and encode the texts.

    Args:
        - texts (list): A list of strings, where each string is a text sample.
        - labels (list): A list of labels, where each label corresponds to a text sample in `texts`.
        - tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing and encoding the texts.

    Returns:
        - pytorch_ds: A PyTorch dataset containing the tokenized and encoded texts and their corresponding labels.

    """
    encodings = tokenizer(texts, truncation=True, padding=True)
    pytorch_ds = FinancialNewsDataset(encodings, labels)

    return pytorch_ds



