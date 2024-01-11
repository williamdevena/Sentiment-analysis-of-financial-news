import logging

import numpy as np
from transformers import pipeline

from utils import metrics
from typing import List


def test_hugging_face_pipeline(model,
                               X_test: List[str],
                               y_test: np.ndarray,
                               path_conf_matrix: str,
                               device: int = -1) -> None:
    """
    Tests a Hugging Face pipeline for sentiment analysis with the given model.

    This function evaluates the model on the test data and logs metrics, including the confusion matrix.

    Args:
        model: The model to be used in the pipeline.
        X_test (List[str]): Test data feature array (text data).
        y_test (np.ndarray): Test data target array (labels).
        path_conf_matrix (str): Path to save the confusion matrix plot.
        device (int): The device to run the model on (-1 for CPU, GPU id otherwise).

    Returns:
        None
    """
    pipe = pipeline("sentiment-analysis",
                    model=model,
                    batch_size=50,
                    device=device)
    y_pred = pipe(list(X_test))
    y_pred = [pred['label'] for pred in y_pred]
    print(np.unique(y_pred, return_counts=True))
    y_pred = transform_string_labels_to_num(y_pred)
    #accuracy = sum(a_ == b_ for a_, b_ in zip(y_pred, y_test))/len(y_test)
    logging.info(f"\nHUGGING FACE PIPELINE")
    metrics.log_metrics(y=y_test, y_pred=y_pred, path_conf_matrix=path_conf_matrix)



def transform_string_labels_to_num(y: List[str]) -> List[int]:
    """
    Transforms string labels to numerical labels.

    Converts 'positive' to 1, 'negative' to -1, and 'neutral' to 0.

    Args:
        y (List[str]): List of string labels.

    Returns:
        List[int]: List of transformed numerical labels.
    """
    y = [1 if label=='positive'
            else -1 if label=='negative'
            else 0
            for label in y]

    return y


