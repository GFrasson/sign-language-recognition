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
        specialist_only_velocity: bool = False,
        balance_specialist_data: bool = False
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
        self.balance_specialist_data = balance_specialist_data

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

    def initialize_maps(self, y: np.ndarray):
        """
        Initializes general label maps based on the provided labels (e.g. y_train).
        Necessary when loading a pretrained model to ensure maps match the training state.
        """
        y_merged = y.copy()
        for src, dst in self.merge_class_map.items():
            y_merged[y_merged == src] = dst
            
        unique_classes = np.unique(y_merged)
        self.general_label_map = {original: new for new, original in enumerate(unique_classes)}
        self.general_inv_label_map = {new: original for new, original in enumerate(unique_classes)}
        
        print(f"HierarchicalModel maps initialized. {len(unique_classes)} general classes.")

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

    def _subsample_specialist_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsamples data for classes that are part of specialist configurations.
        Keeps 50% of data for each specialist class.
        """
        specialist_classes_flat = {c for classes in self.specialist_configs.values() for c in classes}
        
        # If no specialist classes, return original
        if not specialist_classes_flat:
            return X, y

        indices_to_keep = []
        unique_classes = np.unique(y)

        for c in unique_classes:
            c_indices = np.where(y == c)[0]
            
            if c in specialist_classes_flat:
                # Keep 50% of the data
                n_keep = int(len(c_indices) * 0.5)
                if n_keep > 0:
                    keep_idx = np.random.choice(c_indices, n_keep, replace=False)
                    indices_to_keep.extend(keep_idx)
                else:
                    pass
            else:
                indices_to_keep.extend(c_indices)
        
        indices_to_keep = np.array(sorted(indices_to_keep))
        
        if len(indices_to_keep) == 0:
            print("Warning: Balancing resulted in empty dataset. Returning original.")
            return X, y

        print(f"Data Balancing: Reduced General Model training data from {len(y)} to {len(indices_to_keep)}")
        return X[indices_to_keep], y[indices_to_keep]

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
        
        X_train_gen_input, y_train_gen_input = X_train, y_train
        if self.balance_specialist_data:
            X_train_gen_input, y_train_gen_input = self._subsample_specialist_classes(X_train, y_train)

        y_train_gen, n_classes_gen = self._prepare_general_labels(y_train_gen_input)
        y_val_gen, _ = self._prepare_general_labels(y_val)
        
        X_train_gen = self._get_general_features(X_train_gen_input)
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

    def load_model(self, folder: str):
        """Loadds both models from subdirectories."""
        # Load General Model
        # We need to instantiate a dummy Model first or direct load. 
        # Since Model.load_model replaces self.model, we can just instantiate a generic one.
        # However, we need n_features and n_classes.
        # Assuming the folder structure exists, we can try to load.
        
        # General Model
        general_folder = path.join(folder, "general")
        # We assume self.general_model is initialized or we initialize it here?
        # In current logic, self.general_model is None until train_model. 
        # We should accept that we are loading into a "blank" hierarchical model.
        # We need to initialize the sub-models objects first.
        
        # Initialize General Model placeholder (params don't matter as load_model overwrites self.model)
        self.general_model = Model(1, 1, 1) # Dummy params
        self.general_model.load_model(general_folder)
        
        # Load Specialist Models
        for trigger in self.specialist_configs.keys():
            spec_folder = path.join(folder, f"specialist_{trigger}")
            if path.exists(spec_folder):
                spec_model = SpecialistModel(1, 1) # Dummy params
                spec_model.load_model(spec_folder)
                self.specialist_models[trigger] = spec_model
            else:
                print(f"Warning: Specialist model folder not found for trigger {trigger} at {spec_folder}")
