from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    signalers: np.ndarray
    is_augmented: np.ndarray = None
    
    @property
    def n_classes(self) -> int:
        return len(self.unique_classes)

    @property
    def unique_classes(self) -> np.ndarray:
        return np.unique(self.y)

    def get_data(self, selected_signalers: list[int], allow_augmented: bool = True) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isin(self.signalers, selected_signalers)
        
        if not allow_augmented and self.is_augmented is not None:
            mask = mask & (~self.is_augmented)

        return self.X[mask], self.y[mask]

    def split_fold(self, train_signalers: list[int], val_signaler: int, test_signaler: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, y_train = self.get_data(train_signalers, allow_augmented=True)
        X_val, y_val = self.get_data([val_signaler], allow_augmented=True)
        X_test, y_test = self.get_data([test_signaler], allow_augmented=False)

        return X_train, y_train, X_val, y_val, X_test, y_test
