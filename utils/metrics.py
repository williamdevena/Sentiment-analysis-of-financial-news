import logging

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from evaluate.visualization import radar_plot
from sklearn.metrics import confusion_matrix
from typing import Tuple


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
    """
    Computes various evaluation metrics for model predictions.

    Args:
        eval_pred (Tuple[np.ndarray, np.ndarray]): A tuple containing the logits from model predictions and the ground truth labels.

    Returns:
        dict: A dictionary containing computed metrics such as precision, recall, F1-score, and accuracy.
    """
    acc = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    results = {"precision": precision.compute(predictions=predictions, references=labels, average="macro")['precision'],
               "recall": recall.compute(predictions=predictions, references=labels, average="macro")['recall'],
               "f1": f1.compute(predictions=predictions, references=labels, average="macro")['f1'],
               "accuracy": acc.compute(predictions=predictions, references=labels)['accuracy']}

    return results




def log_metrics(y: np.ndarray,
                y_pred: np.ndarray,
                path_conf_matrix: str,
                log: bool = True) -> Tuple[float, float, float, float]:
    """
    Logs metrics and builds a confusion matrix for the predictions.

    Args:
        y (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        path_conf_matrix (str): Path to save the confusion matrix plot.
        log (bool): If True, logs the metrics.

    Returns:
        Tuple[float, float, float, float]: Average accuracy, precision, recall, and F1-score.
    """
    build_and_save_conf_matrix(y_pred=y_pred, y=y, path=path_conf_matrix)
    scores = calculate_metrics(y=y, y_pred=y_pred)
    avg_acc = scores['accuracy']
    avg_recall = np.mean(scores['recall'])
    avg_precision = np.mean(scores['precision'])
    avg_f1 = np.mean(scores['f1_score'])
    if log:
        logging.info(f"AVG ACCURACY: {avg_acc}")
        logging.info(f"AVG PRECISION: {avg_precision}")
        logging.info(f"AVG RECALL: {avg_recall}")
        logging.info(f"AVG F1-SCORE: {avg_f1}")

    return avg_acc, avg_precision, avg_recall, avg_f1



def build_and_save_radar_plot(metrics: dict, path_plot: str) -> None:
    """
    Builds and saves a radar plot for the given metrics.

    Args:
        metrics (dict): A dictionary containing metrics to be plotted.
        path_plot (str): File path to save the radar plot.

    Returns:
        None
    """
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    fig = go.Figure()

    for model, dict_metrics in metrics.items():
        fig.add_trace(go.Scatterpolar(
            r=[v for k, v in dict_metrics.items()],
            theta=categories,
            fill='toself',
            name=model
        ))
    fig.write_image(path_plot)



def build_and_save_conf_matrix(y_pred: np.ndarray, y: np.ndarray, path: str) -> None:
    """
    Builds and saves a confusion matrix from prediction and ground truth data.

    Args:
        y_pred (np.ndarray): Predicted labels.
        y (np.ndarray): Actual labels.
        path (str): File path to save the confusion matrix plot.

    Returns:
        None
    """
    cf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', xticklabels=["neg", "neu", "pos"], yticklabels=["neg", "neu", "pos"])
    if path!=None:
        plt.savefig(path)
        plt.close()

def calculate_metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates accuracy, recall, precision, and F1-score based on true and predicted labels.

    Args:
        y (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.

    Returns:
        dict: Dictionary with accuracy, recall, precision, and F1-score.
    """
    cm = confusion_matrix(y, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm, axis=None)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    f1_score = 2*(recall*precision) / (recall+precision)

    metrics = {'accuracy': accuracy,
               'recall': recall,
               'precision': precision,
               'f1_score': f1_score}

    return metrics