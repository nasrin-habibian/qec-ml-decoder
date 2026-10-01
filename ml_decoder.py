"""Machine-learning decoder."""

from sklearn.linear_model import LogisticRegression


def train_ml_decoder(X_train, y_train, random_state: int = 42):
    """
    Train a multinomial logistic-regression decoder.

    The model learns a mapping from syndrome bits to
    single-error/no-error classes.
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model
