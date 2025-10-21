import os
import tensorflow as tf
from datetime import datetime
from os import makedirs, path, rename
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import build_model, evaluate_model_for_cross_validation, train_model, evaluate_model, save_model
from video_processing import list_video_files, extract_features_and_labels
from geometric_features import NUM_ANGLES_PER_HAND, NUM_POSE_DISTANCES


NUM_FRAMES = 15
N_FEATURES = 2 * NUM_ANGLES_PER_HAND + NUM_POSE_DISTANCES
LSTM_UNITS = 512
DATA_PATH = "data/videos"
FEATURES_PATH = "data/features"
MODELS_PATH = "models"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

# For deterministic behavior on CuDNN backend (GPU)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set a fixed value for the Python hash seed to further aid reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)


def plot_confusion_matrix(y_test, y_pred, unique_labels, title: str, models_folder: str):
    matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=unique_labels)
    disp.plot(cmap=plt.cm.Blues)

    plt.title(title)
    plt.savefig(path.join(models_folder, "confusion_matrix.png"))
    plt.close()

# ==========================
# 2. Preparação dos Dados
# ==========================
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


def cross_validate_leave_two_signalers_out(X, y, signalers):
    """
    Realiza validação cruzada, deixando um sinalizador para validação e outro para teste.
    Os demais são usados para treino. A ordem dos pares é fixa para garantir consistência.
    Calcula a acurácia final considerando todas as predições de teste.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_models_folder = path.join(MODELS_PATH, timestamp)
    makedirs(current_models_folder, exist_ok=True)

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
        train_signalers = [s for s in ordered_signalers if s not in [val_signaler, test_signaler]]

        print(val_signaler)
        print(test_signaler)
        print(train_signalers)

        X_train, y_train = select_by_signalers(X, y, signalers, train_signalers)
        X_val, y_val = select_by_signalers(X, y, signalers, [val_signaler])
        X_test, y_test = select_by_signalers(X, y, signalers, [test_signaler])

        n_classes = len(np.unique(y))
        model = build_model(N_FEATURES, LSTM_UNITS, n_classes)
        history = train_model(model, X_train, y_train, X_val, y_val)

        y_pred, test_acc = evaluate_model_for_cross_validation(model, X_test, y_test)

        # Aggregate predictions and labels
        all_test_preds.extend(y_pred)
        all_test_labels.extend(y_test)

        models_folder = path.join(current_models_folder, f"{i}_fold_lstm_{test_acc:.4f}_val_{val_signaler}_test_{test_signaler}")
    
        save_model(model, models_folder)
        plot_training_history(history, models_folder)
        plot_confusion_matrix(y_test, y_pred, np.unique(y), f"Confusion Matrix - Val: {val_signaler}, Test: {test_signaler}", models_folder)

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
    rename(current_models_folder, f"{current_models_folder}_lstm_{overall_acc:.4f}")
    plot_confusion_matrix(all_test_labels, all_test_preds, np.unique(y), "Confusion Matrix - All Folds", f"{current_models_folder}_lstm_{overall_acc:.4f}")

    return results


def split_dataset(X, y, train_ratio=0.8, val_ratio=0.1):
    """Divide os dados em treino, validação e teste."""
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * (train_ratio + val_ratio))

    return (
        X[:train_size], y[:train_size],
        X[train_size:val_size], y[train_size:val_size],
        X[val_size:], y[val_size:]
    )


def plot_training_history(history, folder="models"):
    """Gera e salva gráficos de acurácia e perda."""
    makedirs(folder, exist_ok=True)
    accuracy_file_path = path.join(folder, "accuracy.png")
    loss_file_path = path.join(folder, "loss.png")

    plt.plot(history.history['accuracy'], label='Acurácia')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.legend()
    plt.savefig(accuracy_file_path)
    plt.close()

    plt.plot(history.history['loss'], label='Perda')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.legend()
    plt.savefig(loss_file_path)
    plt.close()


# ==========================
# Pipeline Principal
# ==========================
def main_train_test_split():
    video_files = list_video_files(DATA_PATH)

    print(f"Encontrados {len(video_files)} vídeos para processar")
    X, y, signalers = extract_features_and_labels(video_files, NUM_FRAMES, FEATURES_PATH)

    if len(X) == 0:
        print("Nenhum vídeo foi processado com sucesso. Verifique os caminhos e formatos dos arquivos.")
        return

    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    print(f"Shape dos dados X: {X.shape}")
    print(f"Shape dos dados y: {y.shape}")
    print(f"Classes únicas encontradas: {unique_classes}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_by_signaler(X, y, signalers)

    print(f"Dados de treino: {len(X_train)} amostras")
    print(f"Dados de validação: {len(X_val)} amostras")
    print(f"Dados de teste: {len(X_test)} amostras")

    model = build_model(N_FEATURES, LSTM_UNITS, n_classes)
    model.summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    history = train_model(model, X_train, y_train, X_val, y_val)
    _, test_acc = evaluate_model(model, X_test, y_test)

    models_folder = path.join("models", f"{timestamp}_lstm_{test_acc:.4f}")
    
    plot_training_history(history, models_folder)
    save_model(model, models_folder)


def main():
    video_files = list_video_files(DATA_PATH)

    print(f"Encontrados {len(video_files)} vídeos para processar")
    X, y, signalers = extract_features_and_labels(video_files, NUM_FRAMES, FEATURES_PATH, augment_factor=0)

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
