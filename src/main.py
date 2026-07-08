from typing import List, Tuple, Dict, Union
from datetime import datetime
from os import path, rename
import random
import os
import argparse
import gc
from glob import glob
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from entities.Dataset import Dataset
from file_utils import make_directories, list_filepaths_with_extension

from entities.Model import Model
from entities.HierarchicalModel import HierarchicalModel
from entities.SpecialistModel import SpecialistModel
from entities.Settings import Settings, GeometricFeaturesSettings, ModelSettings, VelocityFeaturesSettings
from entities.ExperimentConfig import ExperimentConfig
from video_processing import process_videos
from plot import plot_confusion_matrix, plot_training_history


def setup_environment(seed: int) -> np.random.Generator:
    """
    Sets up the random seeds and environment variables for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(e)

    # For deterministic behavior on CuDNN backend (GPU)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Set a fixed value for the Python hash seed to further aid reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    return rng


def create_models_folder(base_path: str) -> str:
    """
    Creates a timestamped folder for saving models and results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_models_folder = path.join(base_path, timestamp)
    make_directories(current_models_folder)

    return current_models_folder


def find_fold_folder(base_path: str, fold_idx: int, val_signaler: int, test_signaler: int) -> str:
    """Finds the existing fold folder matching the criteria."""
    # Pattern: {fold_idx}_fold_lstm_*_val_{val}_test_{test}
    pattern = path.join(base_path, f"{fold_idx}_fold_lstm_*_val_{val_signaler}_test_{test_signaler}")
    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No folder found matching pattern: {pattern}")
    return matches[0]


def _train_specialist_only(dataset, X_train, y_train, X_val, y_val, X_test, y_test, config: ExperimentConfig, load_folder: str = None):
    """Helper to train purely a specialist model."""
    n_features = dataset.X.shape[2] 
    
    if config.specialist_only_velocity:
        X_train = X_train[:, :, GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES:]
        X_val = X_val[:, :, GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES:]
        X_test = X_test[:, :, GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES:]
        n_features = X_train.shape[2]
        print(f"Sliced features for Specialist (Velocity Only). New n_features: {n_features}")

    model = SpecialistModel(n_features, dataset.n_classes)
    
    if load_folder:
        print(f"Loading Specialist Model from {load_folder}")
        model.load_model(load_folder)
        history = None
    else:
        history = model.train_model(X_train, y_train, X_val, y_val)
        
    y_pred = model.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    print(f"Acurácia no conjunto de teste (Specialist Only): {test_acc:.4f}")

    return y_pred, test_acc, history, model


def _train_standard_model(dataset, X_train, y_train, X_val, y_val, X_test, y_test, config: ExperimentConfig, load_folder: str = None):
    """Helper to train standard or hierarchical model."""
    merge_map = {}
    specialist_configs = {}

    if config.use_specialist_4_7:
        merge_map[7] = 4
        specialist_configs[4] = [4, 7]
    
    if config.use_specialist_16_17:
        merge_map[17] = 16
        specialist_configs[16] = [16, 17]

    n_features = GeometricFeaturesSettings.N_FEATURES
    if not specialist_configs:
        if config.general_only_velocity:
            X_train = X_train[:, :, GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES:]
            X_val = X_val[:, :, GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES:]
            X_test = X_test[:, :, GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES:]
            n_features = X_train.shape[2]
            print(f"Sliced features for General Model (Velocity Only). New n_features: {n_features}")
        elif config.general_only_expansion:
            base_geometric_count = (2 * GeometricFeaturesSettings.NUM_ANGLES_PER_HAND) + GeometricFeaturesSettings.NUM_POSE_DISTANCES
            if config.use_velocity:
                X_train = X_train[:, :, base_geometric_count:]
                X_val = X_val[:, :, base_geometric_count:]
                X_test = X_test[:, :, base_geometric_count:]
                print(f"Sliced features for General Model (Expansion + Velocity).")
            else:
                X_train = X_train[:, :, base_geometric_count:GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES]
                X_val = X_val[:, :, base_geometric_count:GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES]
                X_test = X_test[:, :, base_geometric_count:GeometricFeaturesSettings.NUM_GEOMETRIC_FEATURES]
                print(f"Sliced features for General Model (Expansion Only).")
            n_features = X_train.shape[2]
            print(f"New n_features: {n_features}")

    if specialist_configs:
        model = HierarchicalModel(
            merge_map, 
            specialist_configs,
            general_use_velocity=config.use_velocity,
            specialist_only_velocity=config.specialist_only_velocity,
            balance_specialist_data=config.balance_specialist_data
        )
    else:
        model = Model(n_features, Settings.LSTM_UNITS, dataset.n_classes)

    if load_folder:
        print(f"Loading Model from {load_folder}")
        if isinstance(model, HierarchicalModel):
            model.initialize_maps(y_train)
        model.load_model(load_folder)
        history = None
    else:
        history = model.train_model(X_train, y_train, X_val, y_val)
        
    y_pred, test_acc = model.evaluate_model_for_cross_validation(X_test, y_test)

    return y_pred, test_acc, history, model


def train_and_evaluate_fold(dataset: Dataset, ordered_signalers: np.ndarray, val_signaler: int, test_signaler: int, current_models_folder: str, fold_idx: int, config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Trains and evaluates a single fold of the cross-validation.
    """
    train_signalers = [s for s in ordered_signalers if s not in [val_signaler, test_signaler]]

    X_train, y_train, X_val, y_val, X_test, y_test = dataset.split_fold(
        train_signalers,
        val_signaler,
        test_signaler
    )

    print(f"Test split size: {len(X_test)}")
    print(f"Val split size: {len(X_val)}")
    print(f"Train split size: {len(X_train)}")
    
    load_fold_folder = None
    if config.evaluate_mode and config.load_models_from:
        try:
            load_fold_folder = find_fold_folder(config.load_models_from, fold_idx, val_signaler, test_signaler)
            print(f"Found existing fold folder: {load_fold_folder}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    if config.train_specialist_only is not None:
        y_pred, test_acc, history, model = _train_specialist_only(
            dataset, X_train, y_train, X_val, y_val, X_test, y_test, config, load_folder=load_fold_folder
        )
    else:
        y_pred, test_acc, history, model = _train_standard_model(
            dataset, X_train, y_train, X_val, y_val, X_test, y_test, config, load_folder=load_fold_folder
        )

    # Save model and plots for this fold
    if not config.evaluate_mode:
        fold_folder_name = f"{fold_idx}_fold_lstm_{test_acc:.4f}_val_{val_signaler}_test_{test_signaler}"
        models_fold_path = path.join(current_models_folder, fold_folder_name)

        model.save_model(models_fold_path)
        plot_training_history(history, models_fold_path)
        plot_confusion_matrix(
            y_test,
            y_pred,
            dataset.unique_classes,
            f"Confusion Matrix - Val: {val_signaler}, Test: {test_signaler}",
            models_fold_path,
            target_names=dataset.unique_class_names
        )
    # Clean up memory to prevent OOM
    del model
    del history
    tf.keras.backend.clear_session()
    gc.collect()

    return y_pred, y_test, test_acc


def aggregate_and_finalize_results(predictions: List[int], labels: List[int], unique_classes: np.ndarray, unique_class_names: np.ndarray, current_models_folder: str) -> float:
    """
    Calculates overall accuracy, renames the results folder, and plots the final confusion matrix.
    """
    all_test_preds = np.array(predictions)
    all_test_labels = np.array(labels)
    
    overall_acc = accuracy_score(all_test_labels, all_test_preds)
    overall_prec = precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
    overall_rec = recall_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
    
    print("\n" + "="*30)
    print("FINAL GLOBAL METRICS")
    print("="*30)
    print(f"Global Accuracy:  {overall_acc:.4f}")
    print(f"Global Precision: {overall_prec:.4f}")
    print(f"Global Recall:    {overall_rec:.4f}")

    # Rename the folder to include the overall accuracy
    if not path.exists(current_models_folder):
        # In eval mode, if we didn't create folders per fold, check if we need to create base
        make_directories(current_models_folder)
        
    final_folder_path = f"{current_models_folder}_lstm_{overall_acc:.4f}"
    try:
        rename(current_models_folder, final_folder_path)
    except OSError:
        # Fallback if rename fails or folder exists
        final_folder_path = current_models_folder

    plot_confusion_matrix(
        all_test_labels, 
        all_test_preds, 
        unique_classes, 
        "Confusion Matrix - All Folds", 
        final_folder_path,
        target_names=unique_class_names
    )
    
    # Save metrics text
    with open(path.join(final_folder_path, "global_metrics.txt"), "w") as f:
        f.write(f"Global Accuracy: {overall_acc:.4f}\n")
        f.write(f"Global Precision: {overall_prec:.4f}\n")
        f.write(f"Global Recall: {overall_rec:.4f}\n")

    return overall_acc


def save_run_settings(file_path: str, config: ExperimentConfig) -> None:
    """
    Saves the configuration used for the current run to a text file.
    """
    with open(file_path, 'w') as f:
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("========================================\n\n")
        
        f.write("[Arguments]\n")
        f.write(f"use_specialist_4_7: {config.use_specialist_4_7}\n")
        f.write(f"use_specialist_16_17: {config.use_specialist_16_17}\n")
        f.write(f"train_specialist_only: {config.train_specialist_only}\n")
        f.write(f"legacy_features: {config.legacy_features}\n")
        f.write(f"general_only_velocity: {config.general_only_velocity}\n")
        f.write(f"general_only_expansion: {config.general_only_expansion}\n\n")
        
        f.write("[Settings]\n")
        for attr in dir(Settings):
            if not attr.startswith("__"):
                val = getattr(Settings, attr)
                if not callable(val):
                    f.write(f"{attr}: {val}\n")
        f.write("\n")
        
        f.write("[ModelSettings]\n")
        for attr in dir(ModelSettings):
            if not attr.startswith("__"):
                val = getattr(ModelSettings, attr)
                if not callable(val):
                    f.write(f"{attr}: {val}\n")
        f.write("\n")
        
        f.write("[GeometricFeaturesSettings]\n")
        f.write(f"N_FEATURES: {GeometricFeaturesSettings.N_FEATURES}\n")
        f.write(f"USE_LEGACY_FEATURES: {GeometricFeaturesSettings.USE_LEGACY_FEATURES}\n")
        f.write(f"USE_VELOCITY_FEATURES: {GeometricFeaturesSettings.USE_VELOCITY_FEATURES}\n")
        f.write(f"NUM_ANGLES_PER_HAND: {GeometricFeaturesSettings.NUM_ANGLES_PER_HAND}\n")
        f.write(f"NUM_POSE_DISTANCES: {GeometricFeaturesSettings.NUM_POSE_DISTANCES}\n")
        if not GeometricFeaturesSettings.USE_LEGACY_FEATURES:
            f.write(f"NUM_DISTANCES_PER_HAND: {GeometricFeaturesSettings.NUM_DISTANCES_PER_HAND}\n")
            f.write(f"NUM_FACE_ANCHORS: {GeometricFeaturesSettings.NUM_FACE_ANCHORS}\n")
        if GeometricFeaturesSettings.USE_VELOCITY_FEATURES:
            f.write(f"N_VELOCITY_FEATURES: {VelocityFeaturesSettings.N_VELOCITY_FEATURES}\n")


def cross_validate_leave_two_signalers_out(dataset: Dataset, rng: np.random.Generator, config: ExperimentConfig, start_fold: int = 0, resume_folder: str = None) -> Tuple[List[Dict[str, Union[int, float]]], float]:
    """
    Performs Leave-Two-Signalers-Out Cross-Validation.
    """
    if resume_folder and path.exists(resume_folder):
        current_models_folder = resume_folder
        print(f"Resuming execution in folder: {current_models_folder}")
    else:
        current_models_folder = create_models_folder(Settings.MODELS_PATH)
    
    save_run_settings(path.join(current_models_folder, 'run_settings.txt'), config)

    unique_signalers = np.unique(dataset.signalers)
    # unique_signalers = [s for s in unique_signalers if s != 1]

    n_signalers = len(unique_signalers)

    # Fix order for reproducibility using the passed generator
    ordered_signalers = rng.permutation(unique_signalers)

    all_test_predictions = []
    all_test_labels = []
    results = []

    for i in range(start_fold, n_signalers):
        test_signaler = ordered_signalers[i]
        val_signaler = ordered_signalers[(i + 1) % n_signalers]

        y_pred, y_test, test_acc = train_and_evaluate_fold(
            dataset, ordered_signalers, val_signaler, test_signaler, current_models_folder, i, config
        )

        all_test_predictions.extend(y_pred)
        all_test_labels.extend(y_test)

        results.append({
            "val_signaler": val_signaler,
            "test_signaler": test_signaler,
            "test_acc": test_acc
        })
        print(f"Val: {val_signaler}, Test: {test_signaler}, Test accuracy: {test_acc:.4f}")

    overall_acc = aggregate_and_finalize_results(all_test_predictions, all_test_labels, dataset.unique_classes, dataset.unique_class_names, current_models_folder)
    return results, overall_acc


def main():
    parser = argparse.ArgumentParser(description="Sign Language Recognition")
    parser.add_argument('--legacy-features', action='store_true', help='Use legacy 88 features instead of new 102 features')
    parser.add_argument('--use-specialist-4-7', action='store_true', help='Use hierarchical specialist model for classes 4 and 7')
    parser.add_argument('--use-specialist-16-17', action='store_true', help='Use hierarchical specialist model for classes 16 and 17')
    parser.add_argument('--use-velocity', action='store_true', help='Include velocity/acceleration features for temporal dynamics')
    parser.add_argument('--specialist-only-velocity', action='store_true', help='Specialist models use ONLY velocity features (requires at least one specialist flag)')
    parser.add_argument('--train-specialist-only', type=int, default=None, help='Train ONLY the specialist model for the given trigger class (e.g., 4 or 16). Filters data to only relevant classes.')
    parser.add_argument('--balance-specialist-data', action='store_true', help='Use 50% of data for specialist classes when training the General Model to maintain balance.')
    parser.add_argument('--unroll-lstm', action='store_true', help='Unroll LSTM to avoid CuDNN errors with large models')
    parser.add_argument('--start-fold', type=int, default=0, help='Fold index to start execution from (useful for resuming)')
    parser.add_argument('--executions', type=int, default=1, help='Number of times to run the cross-validation with different seeds')
    parser.add_argument('--resume-folder', type=str, default=None, help='Path to existing experiment folder to resume saving results into')
    parser.add_argument('--evaluate-only', action='store_true', help='Skip training and evaluate using models found in --load-models-from')
    parser.add_argument('--load-models-from', type=str, default=None, help='Path to experiment folder containing trained models to load')
    parser.add_argument('--general-only-velocity', action='store_true', help='Use ONLY velocity features for the general model (M11)')
    parser.add_argument('--general-only-expansion', action='store_true', help='Use ONLY expansion features (or Exp + Mov) for the general model (M10, M9)')
    args = parser.parse_args()
    
    if args.evaluate_only and not args.load_models_from:
        parser.error("--evaluate-only requires --load-models-from to be specified")

    config = ExperimentConfig(
        legacy_features=args.legacy_features,
        use_specialist_4_7=args.use_specialist_4_7,
        use_specialist_16_17=args.use_specialist_16_17,
        use_velocity=args.use_velocity,
        specialist_only_velocity=args.specialist_only_velocity,
        train_specialist_only=args.train_specialist_only,
        unroll_lstm=args.unroll_lstm,
        balance_specialist_data=args.balance_specialist_data,
        evaluate_mode=args.evaluate_only,
        load_models_from=args.load_models_from,
        general_only_velocity=args.general_only_velocity,
        general_only_expansion=args.general_only_expansion
    )

    # Configure features based on flags
    GeometricFeaturesSettings.configure(config.legacy_features, config.extraction_use_velocity)
    ModelSettings.LSTM_UNROLL = config.unroll_lstm
    print(f"Feature Mode: {'LEGACY (88)' if config.legacy_features else 'NEW (126)'}")
    print(f"Velocity Features (Extraction): {'ENABLED' if config.extraction_use_velocity else 'DISABLED'}")
    print(f"  - General Model Velocity: {'ENABLED' if config.use_velocity else 'DISABLED'}")
    print(f"  - Specialist Model Velocity: {'ENABLED' if config.specialist_only_velocity else 'DISABLED'}")
    print(f"LSTM Unroll: {'ENABLED' if config.unroll_lstm else 'DISABLED'}")
    print(f"N_FEATURES configured: {GeometricFeaturesSettings.N_FEATURES}")

    rng = setup_environment(Settings.SEED)

    video_files: list[str] = list_filepaths_with_extension(Settings.DATA_PATH, '.mp4')
    print(f"Encontrados {len(video_files)} vídeos para processar")

    dataset = process_videos(video_files, Settings.NUM_FRAMES, Settings.FEATURES_PATH, augment_factor=20, use_legacy=config.legacy_features, use_velocity=config.extraction_use_velocity)

    if len(dataset.X) == 0:
        print("Nenhum vídeo foi processado com sucesso. Verifique os caminhos e formatos dos arquivos.")
        return
    
    print(f"Shape dos dados X: {dataset.X.shape}")
    print(f"Shape dos dados y: {dataset.y.shape}")
    print(f"Classes únicas encontradas: {dataset.unique_classes}")
    print(f"Número de classes: {dataset.n_classes}")

    if config.train_specialist_only is not None:
        trigger = config.train_specialist_only
        print(f"\n[MODE] Training Specialist Model Only (Trigger: {trigger})")
        
        if trigger not in Settings.SPECIALIST_CONFIGS:
            print(f"Error: Unknown specialist trigger {trigger}. Available: {list(Settings.SPECIALIST_CONFIGS.keys())}")
            return

        target_classes = Settings.SPECIALIST_CONFIGS[trigger]
        print(f"Target Original Classes: {target_classes}")

        dataset.filter_by_classes(target_classes)

    all_accuracies = []

    for i in range(args.executions):
        current_seed = Settings.SEED + i
        if args.executions > 1:
            print(f"\n{'='*40}")
            print(f"EXECUTION {i+1}/{args.executions} (Seed: {current_seed})")
            print(f"{'='*40}")
        
        rng = setup_environment(current_seed)
        _, overall_acc = cross_validate_leave_two_signalers_out(dataset, rng, config, start_fold=args.start_fold, resume_folder=args.resume_folder)
        all_accuracies.append(overall_acc)

    if args.executions > 1:
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)
        
        print("\n" + "="*40)
        print("FINAL RESULTS ACROSS ALL EXECUTIONS")
        print("="*40)
        for i, acc in enumerate(all_accuracies):
            print(f"Execution {i+1} Accuracy: {acc:.4f}")
        print(f"Mean Global Accuracy: {mean_acc:.4f}")
        print(f"Standard Deviation: {std_acc:.4f}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = path.join(Settings.MODELS_PATH, f"executions_summary_{timestamp}.txt")
        make_directories(Settings.MODELS_PATH)
        with open(summary_file, "w") as f:
            f.write(f"Total Executions: {args.executions}\n")
            f.write(f"Base Seed: {Settings.SEED}\n\n")
            for i, acc in enumerate(all_accuracies):
                f.write(f"Execution {i+1} (Seed {Settings.SEED + i}) Accuracy: {acc:.4f}\n")
            f.write(f"\nMean Global Accuracy: {mean_acc:.4f}\n")
            f.write(f"Standard Deviation: {std_acc:.4f}\n")
        print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
