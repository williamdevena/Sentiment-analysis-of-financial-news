from transformers import pipeline


def test_hugging_face_pipeline(model, X_test, y_test, labelling_function, device=-1):
    pipe = pipeline("text-classification",
                    model=model,
                    batch_size=50,
                    device=device)
    y_pred = pipe(list(X_test))
    y_pred = [pred['label'] for pred in y_pred]
    y_pred = labelling_function(y_pred)

    #print(np.unique(y_pred, return_counts=True))
    #print(list(y_test))

    accuracy = sum(a_ == b_ for a_, b_ in zip(y_pred, y_test))/len(y_test)

    return accuracy


def labelling_function_twitter_roberta(y):
    y = [1 if label=='LABEL_2'
              else -1 if label=='LABEL_0'
              else 0
              for label in y]

    return y


def labelling_function_financial_bert(y):
    y = [1 if label=='positive'
            else -1 if label=='negative'
            else 0
            for label in y]

    return y