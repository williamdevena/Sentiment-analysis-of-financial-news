import logging

import numpy as np
from transformers import pipeline

from utils import metrics


def test_hugging_face_pipeline(model, X_test, y_test, path_conf_matrix, device=-1):
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
    #print(y_pred)
    metrics.log_metrics(y=y_test, y_pred=y_pred, path_conf_matrix=path_conf_matrix)


# def labelling_function_twitter_roberta(y):
#     y = [1 if label=='LABEL_2'
#               else -1 if label=='LABEL_0'
#               else 0
#               for label in y]

#     return y


def transform_string_labels_to_num(y):
    y = [1 if label=='positive'
            else -1 if label=='negative'
            else 0
            for label in y]

    return y



# def labelling_function_financial_bert(y):
#     y = [1 if label=='positive'
#             else -1 if label=='negative'
#             else 0
#             for label in y]

#     return y