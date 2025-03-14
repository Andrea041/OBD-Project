import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_mini_batches(X_train, y_train, batch_size, shuffle=True):
    # Convert y_train to NumPy array if it's a Pandas Series
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy().reshape(-1, 1)  # Ensure correct shape

    num_samples = X_train.shape[0]

    # Stratified K-Fold to ensure each batch has a balanced class representation
    skf = StratifiedKFold(n_splits=num_samples // batch_size, shuffle=shuffle)

    mini_batches = []
    for _, batch_indices in skf.split(X_train, y_train):
        mini_batch_X = X_train[batch_indices]
        mini_batch_y = y_train[batch_indices]
        mini_batches.append((mini_batch_X, mini_batch_y))

    return mini_batches
