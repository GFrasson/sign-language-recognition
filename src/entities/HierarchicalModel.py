from os import makedirs, path
import numpy as np
from typing import Tuple, List, Dict

from entities.Model import Model
from entities.SpecialistModel import SpecialistModel
from entities.Dataset import Dataset
from entities.Settings import Settings, GeometricFeaturesSettings, ModelSettings

class HierarchicalModel:
    def __init__(self, 
                 merge_class_map: Dict[int, int], 
                 specialist_trigger_class: int, 
                 specialist_classes: List[int]):
        """
        Args:
            merge_class_map: Dict mapping original classes to merged class for General Model.
                             e.g. {7: 4} means class 7 becomes class 4.
            specialist_trigger_class: The class ID in the General Model that triggers the Specialist Model.
                                      e.g. 4 (after merge).
            specialist_classes: The list of original classes the specialist distinguishes between.
                                e.g. [4, 7].
        """
        self.merge_class_map = merge_class_map
        self.specialist_trigger_class = specialist_trigger_class
        self.specialist_classes = sorted(specialist_classes)
        
        # General Model will need to handle remapped labels.
        # We need to determine N classes for General Model.
        # This will be determined dynamically during training/init based on dataset.
        self.general_model: Model = None
        self.specialist_model: Model = None
        
        # Mappings for General Model (Original -> General Label)
        self.general_label_map = {} 
        self.general_inv_label_map = {}
        
        # Mappings for Specialist Model (Original -> Specialist Label)
        self.specialist_label_map = {original: i for i, original in enumerate(self.specialist_classes)}
        self.specialist_inv_label_map = {i: original for i, original in enumerate(self.specialist_classes)}

    def _prepare_general_labels(self, y: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Merges classes and remaps them to a contiguous range 0..N-1.
        """
        y_merged = y.copy()
        for src, dst in self.merge_class_map.items():
            y_merged[y_merged == src] = dst
            
        unique_classes = np.unique(y_merged)
        self.general_label_map = {original: new for new, original in enumerate(unique_classes)}
        self.general_inv_label_map = {new: original for new, original in enumerate(unique_classes)}
        
        y_mapped = np.array([self.general_label_map[label] for label in y_merged])
        return y_mapped, len(unique_classes)

    def _prepare_specialist_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Filters labels to only those relevant for specialist, and maps to 0..M-1.
        """
        # Note: This expects y to be the filtered y containing only specialist classes
        y_mapped = np.array([self.specialist_label_map[label] for label in y])
        return y_mapped

    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Trains both General and Specialist models using the provided training and validation data.
        Matches the signature of Model.train_model.
        """
        # --- Train General Model ---
        print("\n=== Training General Model ===")
        y_train_gen, n_classes_gen = self._prepare_general_labels(y_train)
        y_val_gen, _ = self._prepare_general_labels(y_val)
        
        self.general_model = Model(GeometricFeaturesSettings.N_FEATURES, Settings.LSTM_UNITS, n_classes_gen)
        # Using same method as Model class but calling the underlying object
        history = self.general_model.train_model(X_train, y_train_gen, X_val, y_val_gen)
        
        # --- Train Specialist Model ---
        print("\n=== Training Specialist Model ===")
        # Filter data for specialist classes
        mask_train = np.isin(y_train, self.specialist_classes)
        mask_val = np.isin(y_val, self.specialist_classes)
        
        X_train_spec = X_train[mask_train]
        y_train_spec = y_train[mask_train]
        
        X_val_spec = X_val[mask_val]
        y_val_spec = y_val[mask_val]
        
        if len(X_train_spec) == 0:
            print("Warning: No training data for specialist model!")
            return history

        y_train_spec_mapped = self._prepare_specialist_labels(y_train_spec)
        y_val_spec_mapped = self._prepare_specialist_labels(y_val_spec)
        
        self.specialist_model = SpecialistModel(
            GeometricFeaturesSettings.N_FEATURES, 
            len(self.specialist_classes)
        )
        self.specialist_model.train_model(
            X_train_spec, 
            y_train_spec_mapped, 
            X_val_spec, 
            y_val_spec_mapped
        )
        
        return history

    def evaluate_model_for_cross_validation(self, X_test, y_test):
        """
        Evaluates the hierarchical model and returns predictions and accuracy.
        Matches the signature of Model.evaluate_model_for_cross_validation.
        """
        y_pred = self.predict(X_test)
        test_acc = np.mean(y_pred == y_test)

        print(f"Acurácia no conjunto de teste (Hierárquico): {test_acc:.4f}")

        return y_pred, test_acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 1. Predict with General Model
        gen_probs = self.general_model.model.predict(X, verbose=0)
        gen_preds_mapped = np.argmax(gen_probs, axis=1)
        
        # Map back to "merged" original IDs
        gen_preds_merged_space = np.array([self.general_inv_label_map[p] for p in gen_preds_mapped])
        
        final_preds = gen_preds_merged_space.copy()
        
        # 2. Identify indices that triggered the specialist
        specialist_indices = np.where(gen_preds_merged_space == self.specialist_trigger_class)[0]
        
        if len(specialist_indices) > 0 and self.specialist_model is not None:
            X_spec = X[specialist_indices]
            spec_probs = self.specialist_model.model.predict(X_spec, verbose=0)
            spec_preds_mapped = np.argmax(spec_probs, axis=1)
            
            # Map specialist predictions back to original IDs (4 or 7)
            spec_preds_original = np.array([self.specialist_inv_label_map[p] for p in spec_preds_mapped])
            
            final_preds[specialist_indices] = spec_preds_original
            
        return final_preds

    def save_model(self, folder: str):
        """
        Saves both models to subdirectories.
        Matches the signature of Model.save_model (but folder structure is internal).
        """
        if self.general_model:
            makedirs(path.join(folder, "general"), exist_ok=True)
            self.general_model.save_model(path.join(folder, "general"))
        if self.specialist_model:
            makedirs(path.join(folder, "specialist"), exist_ok=True)
            self.specialist_model.save_model(path.join(folder, "specialist"))
