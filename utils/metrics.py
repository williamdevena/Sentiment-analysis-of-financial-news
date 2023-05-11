import logging

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from evaluate.visualization import radar_plot
from sklearn.metrics import confusion_matrix


def compute_metrics(eval_pred):
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




def log_metrics(y, y_pred, path_conf_matrix):
    build_and_save_conf_matrix(y_pred=y_pred, y=y, path=path_conf_matrix)
    scores = calculate_metrics(y=y, y_pred=y_pred)
    avg_acc = scores['accuracy']
    avg_recall = np.mean(scores['recall'])
    avg_precision = np.mean(scores['precision'])
    avg_f1 = np.mean(scores['f1_score'])
    logging.info(f"AVG ACCURACY: {avg_acc}")
    logging.info(f"AVG PRECISION: {avg_precision}")
    logging.info(f"AVG RECALL: {avg_recall}")
    logging.info(f"AVG F1-SCORE: {avg_f1}")

    return avg_acc, avg_precision, avg_recall, avg_f1



def build_and_save_radar_plot(metrics, path_plot):
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



def build_and_save_conf_matrix(y_pred, y, path):
    """
    Builds and displays a confusion matrix plot based on the provided true and predicted class labels.
    If a file path is provided, the plot is saved as an image.

    Args:
        - y_pred (array): Predicted class labels.
        - y (array): True class labels.
        - path (str): Optional. File path to save the plot as an image.

    Returns: None.
    """
    cf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', xticklabels=["neg", "neu", "pos"], yticklabels=["neg", "neu", "pos"])
    if path!=None:
        plt.savefig(path)
        plt.close()

def calculate_metrics(y, y_pred):
    """
    Calculates accuarcy, recall, precision, and
    F1-score based on the provided true and predicted class labels.

    Args:
        - y (array): True class labels.
        - y_pred (array): Predicted class labels.

    Returns:
        - metrics (dict): A dictionary containing accuracy,
        recall, precision, and F1-score values for each class label.
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