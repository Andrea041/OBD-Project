import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def generate_mini_batches(X_train, y_train, batch_size):
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy().ravel()

    num_samples = X_train.shape[0]

    if num_samples <= batch_size:
        return [(X_train, y_train)]

    mini_batches = []

    stratified_split = StratifiedShuffleSplit(n_splits=num_samples // batch_size, test_size=batch_size, random_state=42)

    for train_idx, batch_idx in stratified_split.split(X_train, y_train):
        mini_batch_X = X_train[batch_idx]
        mini_batch_y = y_train[batch_idx]
        mini_batches.append((mini_batch_X, mini_batch_y))

        # Stop quando abbiamo generato abbastanza batch
        if len(mini_batches) * batch_size >= num_samples:
            break

    return mini_batches
