from typing import Tuple, List, Dict
from os import makedirs, path
import numpy as np

from entities.Model import Model
from entities.SpecialistModel import SpecialistModel
from entities.Settings import Settings, GeometricFeaturesSettings

class HierarchicalModel:
    def __init__(
        self,
        merge_class_map: Dict[int, int], 
        specialist_configs: Dict[int, List[int]],
        general_use_velocity: bool = False,
        specialist_only_velocity: bool = False
    ):
        """
        Args:
            merge_class_map: Dict mapping original classes to merged class for General Model.
            specialist_configs: Dict where key is the trigger class ID and value is list of original classes to distinguish.
            general_use_velocity: Whether general model should use velocity features if available.
            specialist_only_velocity: Whether specialist models should use ONLY velocity features and ignore geometric.
        """
        self.merge_class_map = merge_class_map
        self.specialist_configs = specialist_configs

        self.general_use_velocity = general_use_velocity
        self.specialist_only_velocity = specialist_only_velocity

        self.general_model: Model = None
        self.specialist_models: Dict[int, SpecialistModel] = {}

        # Mappings for General Model (Original -> General Label)
        self.general_label_map = {} 
        self.general_inv_label_map = {}

        # Mappings for Specialist Models (Trigger -> (Original -> Specialist Label))
        self.specialist_label_maps: Dict[int, Dict[int, int]] = {}
        self.specialist_inv_label_maps: Dict[int, Dict[int, int]] = {}

        for trigger, classes in self.specialist_configs.items():
            sorted_classes = sorted(classes)
            self.specialist_label_maps[trigger] = {original: i for i, original in enumerate(sorted_classes)}
            self.specialist_inv_label_maps[trigger] = {i: original for i, original in enumerate(sorted_classes)}

    def _get_general_features(self, X: np.ndarray) -> np.ndarray:
        """Slices X to return features for the General Model."""
        if self.general_use_velocity:
            return X # Use all features (Geo + Vel)
        else:
            # Slice features on axis 2 (Batch, Frames, FEATURES)
            return X[:, :, :GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES]

    def _get_specialist_features(self, X: np.ndarray) -> np.ndarray:
        """Slices X to return features for the Specialist Model."""
        if self.specialist_only_velocity:
            # Slice features on axis 2 (Batch, Frames, FEATURES)
            return X[:, :, GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES:]
        else:
            return X # Use all features

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

    def _prepare_specialist_labels(self, y: np.ndarray, trigger: int) -> np.ndarray:
        """
        Filters labels to only those relevant for the specific specialist, and maps to 0..M-1.
        """
        # Note: This expects y to be the filtered y containing only specialist classes
        label_map = self.specialist_label_maps[trigger]
        y_mapped = np.array([label_map[label] for label in y])
        return y_mapped

    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Trains General and all configured Specialist models.
        """
        # --- Train General Model ---
        print("\n=== Training General Model ===")
        y_train_gen, n_classes_gen = self._prepare_general_labels(y_train)
        y_val_gen, _ = self._prepare_general_labels(y_val)
        
        X_train_gen = self._get_general_features(X_train)
        X_val_gen = self._get_general_features(X_val)
        
        self.general_model = Model(X_train_gen.shape[2], Settings.LSTM_UNITS, n_classes_gen)
        # Using same method as Model class but calling the underlying object
        history = self.general_model.train_model(X_train_gen, y_train_gen, X_val_gen, y_val_gen)
        
        # --- Train Specialist Models ---
        for trigger, classes in self.specialist_configs.items():
            print(f"\n=== Training Specialist Model (Trigger: {trigger}, Classes: {classes}) ===")
            
            # Filter data for specialist classes
            mask_train = np.isin(y_train, classes)
            mask_val = np.isin(y_val, classes)
            
            X_train_spec = self._get_specialist_features(X_train[mask_train])
            y_train_spec = y_train[mask_train]
            
            X_val_spec = self._get_specialist_features(X_val[mask_val])
            y_val_spec = y_val[mask_val]
            
            if len(X_train_spec) == 0:
                print(f"Warning: No training data for specialist model (Trigger: {trigger})!")
                continue

            y_train_spec_mapped = self._prepare_specialist_labels(y_train_spec, trigger)
            y_val_spec_mapped = self._prepare_specialist_labels(y_val_spec, trigger)
            
            spec_model = SpecialistModel(
                X_train_spec.shape[2], 
                len(classes)
            )
            spec_model.train_model(
                X_train_spec, 
                y_train_spec_mapped, 
                X_val_spec, 
                y_val_spec_mapped
            )
            self.specialist_models[trigger] = spec_model
        
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
        X_gen = self._get_general_features(X)
        gen_probs = self.general_model.model.predict(X_gen, verbose=0)
        gen_preds_mapped = np.argmax(gen_probs, axis=1)
        
        # Map back to "merged" original IDs
        gen_preds_merged_space = np.array([self.general_inv_label_map[p] for p in gen_preds_mapped])
        
        final_preds = gen_preds_merged_space.copy()
        
        # 2. Identify indices that triggered any specialist
        for trigger, spec_model in self.specialist_models.items():
            specialist_indices = np.where(gen_preds_merged_space == trigger)[0]
            
            if len(specialist_indices) > 0:
                X_spec_full = X[specialist_indices]
                X_spec = self._get_specialist_features(X_spec_full)
                
                spec_probs = spec_model.model.predict(X_spec, verbose=0)
                spec_preds_mapped = np.argmax(spec_probs, axis=1)
                
                # Map specialist predictions back to original IDs
                inv_map = self.specialist_inv_label_maps[trigger]
                spec_preds_original = np.array([inv_map[p] for p in spec_preds_mapped])
                
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
        
        for trigger, spec_model in self.specialist_models.items():
            spec_folder = path.join(folder, f"specialist_{trigger}")
            makedirs(spec_folder, exist_ok=True)
            spec_model.save_model(spec_folder)
