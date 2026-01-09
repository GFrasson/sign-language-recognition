# from keras.src.saving.saving_api import load_model
from os import makedirs, path
from keras.models import Sequential
from keras.layers import InputLayer, Normalization, LSTM, Dropout, Dense
from keras.optimizers import AdamW
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf

from entities.Settings import Settings, ModelSettings


class Model:
    def __init__(self, n_features, n_neurons, n_classes, dropout_rate=None, weight_decay=None):
        self.n_features = n_features
        self.n_neurons = n_neurons
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate if dropout_rate is not None else ModelSettings.DROPOUT_RATE
        self.weight_decay = weight_decay if weight_decay is not None else ModelSettings.WEIGHT_DECAY
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Camada de Entrada
        model.add(InputLayer(input_shape=(Settings.NUM_FRAMES, self.n_features)))

        # Camada de Máscara (ignora passos de tempo com padding de zeros)
        # model.add(Masking(mask_value=0.0))

        # Camada de Normalização (importante para features em escalas diferentes)
        model.add(Normalization())

        # Camada Recorrente LSTM
        model.add(LSTM(
            self.n_neurons,
            kernel_regularizer=l1(0.001),  # Regularização L1
            unroll=ModelSettings.LSTM_UNROLL
        ))

        # Camada de Dropout para regularização
        model.add(Dropout(self.dropout_rate))

        # Camada de Classificação
        model.add(Dense(self.n_classes, activation='softmax'))

        # Otimizador AdamW e Função de Perda
        optimizer = AdamW(
            learning_rate=ModelSettings.LEARNING_RATE,
            weight_decay=self.weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        return model

    def train_model(self, X_train, y_train, X_val, y_val, batch_size=None, patience=None):
        """Treina o modelo com early stopping e retorna o histórico."""
        
        actual_batch_size = batch_size if batch_size is not None else ModelSettings.BATCH_SIZE
        actual_patience = patience if patience is not None else ModelSettings.EARLY_STOPPING_PATIENCE

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=actual_patience,
            restore_best_weights=True,
            # start_from_epoch=150
        )

        print("\nIniciando o treinamento do modelo...")
        print(f"Batch Size: {actual_batch_size}, Patience: {actual_patience}")

        tf.debugging.set_log_device_placement(True)
        print("Devices:", tf.config.list_physical_devices())
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=ModelSettings.EPOCHS,
            callbacks=[early_stopping],
            batch_size=actual_batch_size,
        )

    def evaluate_model(self, X_test, y_test):
        """Avalia o modelo no conjunto de teste."""
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"\nAcurácia no conjunto de teste: {test_acc:.4f}")

        return test_loss, test_acc

    def evaluate_model_for_cross_validation(self, X_test, y_test):
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        test_acc = np.mean(y_pred == y_test)

        print(f"Acurácia no conjunto de teste: {test_acc:.4f}")

        return y_pred, test_acc

    def save_model(self, folder="models"):
        """Salva o modelo no formato Keras na pasta especificada."""
        makedirs(folder, exist_ok=True)

        keras_path = path.join(folder, "model.keras")

        self.model.save(keras_path)
        print(f"Modelo salvo em formato .keras: {keras_path}")

    def load_model(self, folder):
        """Carrega o modelo salvo do formato Keras."""
        keras_path = path.join(folder, "model.keras")
        if not path.exists(keras_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {keras_path}")
        
        self.model = tf.keras.models.load_model(keras_path)
        print(f"Modelo carregado de: {keras_path}")

    def summary(self):
        self.model.summary()
