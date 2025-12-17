import numpy as np


def select_by_signalers(X, y, signalers, selected_signalers):
    mask = np.isin(signalers, selected_signalers)
    return X[mask], y[mask]


def split_fold_dataset(X, y, signalers, train_signalers: list[int], val_signaler: int, test_signaler: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = select_by_signalers(X, y, signalers, train_signalers)
    X_val, y_val = select_by_signalers(X, y, signalers, [val_signaler])
    X_test, y_test = select_by_signalers(X, y, signalers, [test_signaler])

    return X_train, y_train, X_val, y_val, X_test, y_test
