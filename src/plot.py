from os import path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from file_utils import make_directories


def plot_confusion_matrix(y_test, y_pred, unique_labels, title: str, models_folder: str):
    matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=unique_labels)
    disp.plot(cmap=plt.cm.Blues)

    plt.title(title)
    plt.savefig(path.join(models_folder, "confusion_matrix.png"))
    plt.close()


def plot_training_history(history, folder="models"):
    """Gera e salva gráficos de acurácia e perda."""
    make_directories(folder)
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
