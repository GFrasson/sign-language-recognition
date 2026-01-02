from os import makedirs, path
from keras.models import Sequential
from keras.layers import InputLayer, Normalization, LSTM, Dropout, Dense
from keras.optimizers import AdamW
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
import numpy as np

from entities.Settings import Settings, ModelSettings


class SpecialistModel:
    """
    Specialized model for distinguishing between similar classes.
    Uses a deeper architecture (2 LSTM layers) with lower learning rate
    to capture fine-grained differences.
    """

    def __init__(
        self,
        n_features,
        n_classes,
        lstm_units_1=None,
        lstm_units_2=None,
        dropout_rate=None,
        weight_decay=None,
        learning_rate=None
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Use specialist-specific defaults
        self.lstm_units_1 = lstm_units_1 or ModelSettings.SPECIALIST_LSTM_UNITS
        self.lstm_units_2 = lstm_units_2 or ModelSettings.SPECIALIST_LSTM_UNITS_2
        self.dropout_rate = dropout_rate or ModelSettings.SPECIALIST_DROPOUT_RATE
        self.weight_decay = weight_decay or ModelSettings.SPECIALIST_WEIGHT_DECAY
        self.learning_rate = learning_rate or ModelSettings.SPECIALIST_LEARNING_RATE
        
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Input Layer
        model.add(InputLayer(input_shape=(Settings.NUM_FRAMES, self.n_features)))

        # Normalization Layer
        model.add(Normalization())

        # First LSTM Layer - returns sequences for the second layer
        model.add(LSTM(
            self.lstm_units_1,
            # return_sequences=True,
            kernel_regularizer=l1(0.001),
        ))

        # Dropout between LSTM layers
        # model.add(Dropout(self.dropout_rate * 0.5))  # Lower dropout between layers

        # Second LSTM Layer - processes the sequences from the first layer
        # model.add(LSTM(
        #     self.lstm_units_2,
        #     kernel_regularizer=l1(0.001),
        # ))

        # Dropout before classification
        model.add(Dropout(self.dropout_rate))

        # Classification Layer
        model.add(Dense(self.n_classes, activation='softmax'))

        # AdamW optimizer with lower learning rate for fine-tuning
        optimizer = AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        return model

    def train_model(self, X_train, y_train, X_val, y_val, batch_size=None, patience=None):
        """Trains the specialist model with early stopping."""
        
        actual_batch_size = batch_size or ModelSettings.SPECIALIST_BATCH_SIZE
        actual_patience = patience or ModelSettings.SPECIALIST_EARLY_STOPPING_PATIENCE

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=actual_patience,
            restore_best_weights=True,
        )

        print("\nSpecialist Model Configuration:")
        print(f"  LSTM Layer 1: {self.lstm_units_1} units")
        print(f"  LSTM Layer 2: {self.lstm_units_2} units")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Batch Size: {actual_batch_size}, Patience: {actual_patience}")

        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=ModelSettings.EPOCHS,
            callbacks=[early_stopping],
            batch_size=actual_batch_size,
        )

    def predict(self, X):
        """Returns class predictions."""
        return np.argmax(self.model.predict(X, verbose=0), axis=1)

    def predict_proba(self, X):
        """Returns class probabilities."""
        return self.model.predict(X, verbose=0)

    def save_model(self, folder="models"):
        """Saves the model in Keras format."""
        makedirs(folder, exist_ok=True)

        keras_path = path.join(folder, "model.keras")

        self.model.save(keras_path)
        print(f"Specialist model saved: {keras_path}")

    def summary(self):
        self.model.summary()
