from os import path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from file_utils import make_directories


def plot_confusion_matrix(y_test, y_pred, unique_labels, title: str, models_folder: str, target_names: list[str] = None):
    matrix = confusion_matrix(y_test, y_pred, labels=unique_labels, normalize='true')

    display_labels = target_names if target_names is not None else unique_labels
    
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='.1%', colorbar=False, text_kw={'fontsize': 9})

    plt.tight_layout()
    plt.savefig(path.join(models_folder, "confusion_matrix.svg"), bbox_inches='tight')
    plt.close('all')


def plot_training_history(history, folder="models"):
    """Gera e salva gráficos de acurácia e perda."""
    make_directories(folder)
    accuracy_file_path = path.join(folder, "accuracy.svg")
    loss_file_path = path.join(folder, "loss.svg")

    plt.plot(history.history['accuracy'], label='Acurácia')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.legend()
    plt.savefig(accuracy_file_path)
    plt.close('all')

    plt.plot(history.history['loss'], label='Perda')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.legend()
    plt.savefig(loss_file_path)
    plt.close('all')
