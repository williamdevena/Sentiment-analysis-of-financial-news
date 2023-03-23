from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier


def train_and_test_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Trains and tests a Naive Bayes classifier.

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
    """
    model_naive = MultinomialNB().fit(X_train, y_train)
    #predicted_naive = model_naive.predict(X_test)
    score = model_naive.score(X_test, y_test)

    return score
