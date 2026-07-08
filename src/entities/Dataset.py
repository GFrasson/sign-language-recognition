from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    signalers: np.ndarray
    class_map: dict[int, str]
    is_augmented: np.ndarray = None
    
    @property
    def n_classes(self) -> int:
        return len(self.unique_classes)

    @property
    def unique_classes(self) -> np.ndarray:
        return np.unique(self.y)

    @property
    def unique_class_names(self) -> np.ndarray:
        """Returns class names sorted by their corresponding label ID."""
        return np.array([self.class_map[label] for label in self.unique_classes])

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

    def filter_by_classes(self, target_classes: list[int]) -> None:
        """
        Filters the dataset to keep only the specified classes and remaps them to 0..N-1.
        Updates X, y, signalers, is_augmented, and class_map in place.
        """
        print(f"Filtering dataset for classes: {target_classes}")
        
        mask = np.isin(self.y, target_classes)
        self.X = self.X[mask]
        self.y = self.y[mask]
        self.signalers = self.signalers[mask]
        if self.is_augmented is not None:
            self.is_augmented = self.is_augmented[mask]
            
        print(f"Filtered Dataset Size: {len(self.X)}")

        # Remap classes to 0..N
        sorted_classes = sorted(target_classes)
        new_class_map = {}
        y_new = np.copy(self.y)
        
        for new_label, original_label in enumerate(sorted_classes):
            y_new[self.y == original_label] = new_label
            
            # Update map: New Label -> Original name
            if original_label in self.class_map:
                new_class_map[new_label] = self.class_map[original_label]
            else:
                new_class_map[new_label] = f"Class {original_label}"

        self.y = y_new
        self.class_map = new_class_map
        print(f"Remapped Labels: {np.unique(self.y)}")
        print(f"New Class Map: {self.class_map}")
