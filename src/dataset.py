import numpy as np


def select_by_signalers(X, y, signalers, selected_signalers):
    mask = np.isin(signalers, selected_signalers)
    return X[mask], y[mask]


def split_dataset_by_signaler(X, y, signalers, train_ratio=0.7, val_ratio=0.15):
    """
    Divide os dados em treino, validação e teste, garantindo que cada sinalizador
    só apareça em um dos conjuntos.
    - X: array de features
    - y: array de labels
    - signalers: lista com o sinalizador de cada amostra (mesma ordem de X/y)
    """
    unique_signalers = np.unique(signalers)
    np.random.shuffle(unique_signalers)

    n_total = len(unique_signalers)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_signalers = unique_signalers[:n_train]
    val_signalers = unique_signalers[n_train:n_train + n_val]
    test_signalers = unique_signalers[n_train + n_val:]

    print(train_signalers)
    print(val_signalers)
    print(test_signalers)

    X_train, y_train = select_by_signalers(X, y, signalers, train_signalers)
    X_val, y_val = select_by_signalers(X, y, signalers, val_signalers)
    X_test, y_test = select_by_signalers(X, y, signalers, test_signalers)

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_dataset(X, y, train_ratio=0.8, val_ratio=0.1):
    """Divide os dados em treino, validação e teste."""
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * (train_ratio + val_ratio))

    return (
        X[:train_size], y[:train_size],
        X[train_size:val_size], y[train_size:val_size],
        X[val_size:], y[val_size:]
    )
