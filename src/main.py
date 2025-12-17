from datetime import datetime
from os import path, rename
import random
import os
import tensorflow as tf
import numpy as np
from dataset import select_by_signalers
from file_utils import make_directories, list_filepaths_with_extension

from entities.Model import Model
from entities.Settings import Settings, GeometricFeaturesSettings
from video_processing import process_videos
from plot import plot_confusion_matrix, plot_training_history


random.seed(Settings.SEED)
np.random.seed(Settings.SEED)
rng = np.random.default_rng(Settings.SEED)
tf.random.set_seed(Settings.SEED)

# For deterministic behavior on CuDNN backend (GPU)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set a fixed value for the Python hash seed to further aid reproducibility
os.environ["PYTHONHASHSEED"] = str(Settings.SEED)

# ==========================
# 2. Preparação dos Dados
# ==========================
def cross_validate_leave_two_signalers_out(X, y, signalers):
    """
    Realiza validação cruzada, deixando um sinalizador para validação e outro para teste.
    Os demais são usados para treino. A ordem dos pares é fixa para garantir consistência.
    Calcula a acurácia final considerando todas as predições de teste.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_models_folder = path.join(Settings.MODELS_PATH, timestamp)
    make_directories(current_models_folder)

    unique_signalers = np.unique(signalers)
    n_signalers = len(unique_signalers)
    results = []

    # Fix order for reproducibility
    ordered_signalers = rng.permutation(unique_signalers)

    all_test_preds = []
    all_test_labels = []

    for i in range(n_signalers):
        test_signaler = ordered_signalers[i]
        val_signaler = ordered_signalers[i + 1 if i + 1 < n_signalers else 0]
        train_signalers = [s for s in ordered_signalers if s not in [
            val_signaler, test_signaler]]

        print(val_signaler)
        print(test_signaler)
        print(train_signalers)

        X_train, y_train = select_by_signalers(
            X, y, signalers, train_signalers)
        X_val, y_val = select_by_signalers(X, y, signalers, [val_signaler])
        X_test, y_test = select_by_signalers(X, y, signalers, [test_signaler])

        n_classes = len(np.unique(y))
        model = Model(GeometricFeaturesSettings.N_FEATURES,
                      Settings.LSTM_UNITS, n_classes)
        history = model.train_model(X_train, y_train, X_val, y_val)

        y_pred, test_acc = model.evaluate_model_for_cross_validation(
            X_test, y_test)

        # Aggregate predictions and labels
        all_test_preds.extend(y_pred)
        all_test_labels.extend(y_test)

        models_folder = path.join(
            current_models_folder, f"{i}_fold_lstm_{test_acc:.4f}_val_{val_signaler}_test_{test_signaler}")

        model.save_model(models_folder)
        plot_training_history(history, models_folder)
        plot_confusion_matrix(y_test, y_pred, np.unique(
            y), f"Confusion Matrix - Val: {val_signaler}, Test: {test_signaler}", models_folder)

        results.append({
            "val_signaler": val_signaler,
            "test_signaler": test_signaler,
            "test_acc": test_acc
        })
        print(f"Val: {val_signaler}, Test: {test_signaler}, Test accuracy: {test_acc:.4f}")

    # Calculate overall accuracy using all predictions
    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)
    overall_acc = np.mean(all_test_preds == all_test_labels)
    print(f"Acurácia final agregada: {overall_acc:.4f}")

    # Update the folder name to save the accuracy
    rename(current_models_folder,
           f"{current_models_folder}_lstm_{overall_acc:.4f}")
    plot_confusion_matrix(all_test_labels, all_test_preds, np.unique(
        y), "Confusion Matrix - All Folds", f"{current_models_folder}_lstm_{overall_acc:.4f}")

    return results


def main():
    video_files: list[str] = list_filepaths_with_extension(Settings.DATA_PATH, '.mp4')

    print(f"Encontrados {len(video_files)} vídeos para processar")
    X, y, signalers = process_videos(video_files, Settings.NUM_FRAMES, Settings.FEATURES_PATH)

    if len(X) == 0:
        print("Nenhum vídeo foi processado com sucesso. Verifique os caminhos e formatos dos arquivos.")
        return

    unique_classes = np.unique(y)

    print(f"Shape dos dados X: {X.shape}")
    print(f"Shape dos dados y: {y.shape}")
    print(f"Classes únicas encontradas: {unique_classes}")

    cross_validate_leave_two_signalers_out(X, y, signalers)


if __name__ == '__main__':
    main()
