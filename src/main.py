import numpy as np
from matplotlib import pyplot as plt

from model import build_model, train_model, evaluate_model, save_model
from video_processing import list_video_files, extract_features_and_labels


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
    NUM_FRAMES = 15
    N_FEATURES = 90
    LSTM_UNITS = 512
    DATA_PATH = "data/videos"

    class_mapping = get_class_mapping()
    video_files = list_video_files(DATA_PATH)

    print(f"Encontrados {len(video_files)} vídeos para processar")
    X, y = extract_features_and_labels(video_files, class_mapping, NUM_FRAMES)

    if len(X) == 0:
        print("Nenhum vídeo foi processado com sucesso. Verifique os caminhos e formatos dos arquivos.")
        return

    print(f"Shape dos dados X: {X.shape}")
    print(f"Shape dos dados y: {y.shape}")
    print(f"Classes únicas encontradas: {np.unique(y)}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
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
