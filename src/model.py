from os import makedirs, path
from keras.models import Sequential
from keras.layers import InputLayer, Masking, Normalization, LSTM, Dropout, Dense
from keras.optimizers import AdamW
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, n_features, n_neurons, n_classes):
        self.n_features = n_features
        self.n_neurons = n_neurons
        self.n_classes = n_classes
        self.model = self.build_model()


    def build_model(self):
        model = Sequential()

        # Camada de Entrada
        model.add(InputLayer(input_shape=(15, self.n_features)))

        # Camada de Máscara (ignora passos de tempo com padding de zeros)
        # model.add(Masking(mask_value=0.0))

        # Camada de Normalização (importante para features em escalas diferentes)
        model.add(Normalization())

        # Camada Recorrente LSTM
        model.add(LSTM(
            self.n_neurons,
            kernel_regularizer=l1(0.001),  # Regularização L1
        ))

        # Camada de Dropout para regularização
        model.add(Dropout(0.4))

        # Camada de Classificação
        model.add(Dense(self.n_classes, activation='softmax'))

        # Otimizador AdamW e Função de Perda
        optimizer = AdamW(
            learning_rate=0.0001,
            weight_decay=0.005
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        return model

    def train_model(self, X_train, y_train, X_val, y_val):
        """Treina o modelo com early stopping e retorna o histórico."""
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            # start_from_epoch=150
        )

        print("\nIniciando o treinamento do modelo...")

        tf.debugging.set_log_device_placement(True)
        print("Devices:", tf.config.list_physical_devices())
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            callbacks=[early_stopping],
            batch_size=1024,
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

    def summary(self):
        self.model.summary()
