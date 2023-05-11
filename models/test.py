import numpy as np


def evaluate_model(model_trainer, test_ds):
    results = model_trainer.evaluate(test_ds)
    metrics = {'ACCURACY': results['eval_accuracy'],
            'PRECISION': results['eval_precision'],
            'RECALL': results['eval_recall'],
            'F1-SCORE': results['eval_f1'],
            }

    y_pred = model_trainer.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred, metrics