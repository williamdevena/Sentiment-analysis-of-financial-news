import numpy as np
from transformers import (AutoModelForSequenceClassification, Trainer,
                          TrainingArguments)

from utils import metrics


def evaluate_model(model_path, test_ds, fun_compute_metrics, path_cm):
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