import numpy as np
from transformers import (AutoModelForSequenceClassification, Trainer)

from utils import metrics
from typing import Tuple
from ..src import pytorch_dataset


def evaluate_model(model_path: str,
                   test_ds: pytorch_dataset.FinancialNewsDataset,
                   fun_compute_metrics: callable,
                   path_cm: str) -> Tuple[np.ndarray, dict]:
    """
    Evaluates the performance of a specified model on a test dataset.

    Loads a model from a given path, evaluates it on the test dataset, and computes various metrics.
    It also builds and saves a confusion matrix.

    Args:
        model_path (str): The file path to the pre-trained model.
        test_ds (Dataset): The test dataset to be evaluated.
        fun_compute_metrics (callable): A function to compute evaluation metrics.
        path_cm (str): File path to save the confusion matrix plot.

    Returns:
        Tuple[np.ndarray, dict]: Predicted labels and a dictionary containing evaluation metrics.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model_trainer = Trainer(
        model=model,
        compute_metrics=fun_compute_metrics,
    )
    results = model_trainer.evaluate(test_ds)
    metrics_dict = {'Accuracy': results['eval_accuracy'],
            'Precision': results['eval_precision'],
            'Recall': results['eval_recall'],
            'F1-score': results['eval_f1'],
            }
    y_pred = model_trainer.predict(test_ds)[0]
    y_pred = np.argmax(y_pred, axis=1)

    metrics.build_and_save_conf_matrix(y_pred=y_pred,
                                       y=test_ds.labels,
                                       path=path_cm)

    return y_pred, metrics_dict