import numpy as np
from matplotlib import pyplot as plt

from model import build_model, train_model, evaluate_model, save_model
from video_processing import list_video_files, extract_features_and_labels
from geometric_features import NUM_ANGLES_PER_HAND, NUM_POSE_DISTANCES


NUM_FRAMES = 15
N_FEATURES = 2 * NUM_ANGLES_PER_HAND + NUM_POSE_DISTANCES
LSTM_UNITS = 512
DATA_PATH = "data/videos"

# ==========================
# 1. Carregamento de Dados
# ==========================
def get_class_mapping():
    """Retorna o mapeamento de classes baseado nas pastas."""
    return {
        '01-professor': 0,
        '02-estudar': 1,
        '03-aluno': 2,
        '04-duvida': 3,
        '05-perguntar': 4,
        '06-responder': 5,
        '07-entender': 6
    }


# ==========================
# 2. Preparação dos Dados
# ==========================
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

    def select_by_signalers(X, y, signalers, selected_signalers):
        mask = np.isin(signalers, selected_signalers)
        return X[mask], y[mask]

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


def plot_training_history(history):
    """Gera e salva gráficos de acurácia e perda."""
    plt.plot(history.history['accuracy'], label='Acurácia')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.close()

    plt.plot(history.history['loss'], label='Perda')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()


# ==========================
# Pipeline Principal
# ==========================
def main():
    class_mapping = get_class_mapping()
    video_files = list_video_files(DATA_PATH)

    print(f"Encontrados {len(video_files)} vídeos para processar")
    X, y, signalers = extract_features_and_labels(video_files, NUM_FRAMES)

    if len(X) == 0:
        print("Nenhum vídeo foi processado com sucesso. Verifique os caminhos e formatos dos arquivos.")
        return

    print(f"Shape dos dados X: {X.shape}")
    print(f"Shape dos dados y: {y.shape}")
    print(f"Classes únicas encontradas: {np.unique(y)}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_by_signaler(X, y, signalers)
    print(f"Dados de treino: {len(X_train)} amostras")
    print(f"Dados de validação: {len(X_val)} amostras")
    print(f"Dados de teste: {len(X_test)} amostras")

    model = build_model(N_FEATURES, LSTM_UNITS, len(class_mapping))
    model.summary()

    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_training_history(history)
    evaluate_model(model, X_test, y_test)
    save_model(model)


if __name__ == '__main__':
    main()
