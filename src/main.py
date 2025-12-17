from typing import List, Tuple, Dict, Union
from datetime import datetime
from os import path, rename
import random
import os
import tensorflow as tf
import numpy as np

from dataset import Dataset
from file_utils import make_directories, list_filepaths_with_extension

from entities.Model import Model
from entities.Settings import Settings, GeometricFeaturesSettings
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


def train_and_evaluate_fold(dataset: Dataset, ordered_signalers: np.ndarray, val_signaler: int, test_signaler: int, current_models_folder: str, fold_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
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

    model = Model(GeometricFeaturesSettings.N_FEATURES, Settings.LSTM_UNITS, dataset.n_classes)

    history = model.train_model(X_train, y_train, X_val, y_val)
    y_pred, test_acc = model.evaluate_model_for_cross_validation(X_test, y_test)

    # Save model and plots for this fold
    fold_folder_name = f"{fold_idx}_fold_lstm_{test_acc:.4f}_val_{val_signaler}_test_{test_signaler}"
    models_fold_path = path.join(current_models_folder, fold_folder_name)

    model.save_model(models_fold_path)
    plot_training_history(history, models_fold_path)
    plot_confusion_matrix(
        y_test,
        y_pred,
        dataset.unique_classes,
        f"Confusion Matrix - Val: {val_signaler}, Test: {test_signaler}",
        models_fold_path
    )

    return y_pred, y_test, test_acc


def aggregate_and_finalize_results(predictions: List[int], labels: List[int], unique_classes: np.ndarray, current_models_folder: str) -> None:
    """
    Calculates overall accuracy, renames the results folder, and plots the final confusion matrix.
    """
    all_test_preds = np.array(predictions)
    all_test_labels = np.array(labels)

    overall_acc = np.mean(all_test_preds == all_test_labels)
    print(f"Acurácia final agregada: {overall_acc:.4f}")

    # Rename the folder to include the overall accuracy
    final_folder_path = f"{current_models_folder}_lstm_{overall_acc:.4f}"
    rename(current_models_folder, final_folder_path)

    plot_confusion_matrix(
        all_test_labels, 
        all_test_preds, 
        unique_classes, 
        "Confusion Matrix - All Folds", 
        final_folder_path
    )


def cross_validate_leave_two_signalers_out(dataset: Dataset, rng: np.random.Generator) -> List[Dict[str, Union[int, float]]]:
    """
    Performs Leave-Two-Signalers-Out Cross-Validation.
    """
    current_models_folder = create_models_folder(Settings.MODELS_PATH)

    unique_signalers = np.unique(dataset.signalers)
    n_signalers = len(unique_signalers)

    # Fix order for reproducibility using the passed generator
    ordered_signalers = rng.permutation(unique_signalers)

    all_test_predictions = []
    all_test_labels = []
    results = []

    for i in range(n_signalers):
        test_signaler = ordered_signalers[i]
        val_signaler = ordered_signalers[(i + 1) % n_signalers]

        y_pred, y_test, test_acc = train_and_evaluate_fold(
            dataset, ordered_signalers, val_signaler, test_signaler, current_models_folder, i
        )

        all_test_predictions.extend(y_pred)
        all_test_labels.extend(y_test)

        results.append({
            "val_signaler": val_signaler,
            "test_signaler": test_signaler,
            "test_acc": test_acc
        })
        print(f"Val: {val_signaler}, Test: {test_signaler}, Test accuracy: {test_acc:.4f}")

    aggregate_and_finalize_results(all_test_predictions, all_test_labels, dataset.unique_classes, current_models_folder)
    return results


def main():
    rng = setup_environment(Settings.SEED)

    video_files: list[str] = list_filepaths_with_extension(Settings.DATA_PATH, '.mp4')
    print(f"Encontrados {len(video_files)} vídeos para processar")

    dataset = process_videos(video_files, Settings.NUM_FRAMES, Settings.FEATURES_PATH)

    if len(dataset.X) == 0:
        print("Nenhum vídeo foi processado com sucesso. Verifique os caminhos e formatos dos arquivos.")
        return
    
    print(f"Shape dos dados X: {dataset.X.shape}")
    print(f"Shape dos dados y: {dataset.y.shape}")
    print(f"Classes únicas encontradas: {dataset.unique_classes}")
    print(f"Número de classes: {dataset.n_classes}")

    cross_validate_leave_two_signalers_out(dataset, rng)


if __name__ == '__main__':
    main()
