from keras.models import Sequential
from keras.layers import InputLayer, Masking, Normalization, LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping



def build_model(n_features, n_neurons, n_classes):
    model = Sequential()

    # Camada de Entrada
    model.add(InputLayer(input_shape=(15, n_features)))

    # Camada de Máscara (ignora passos de tempo com padding de zeros)
    model.add(Masking(mask_value=0.0))

    # Camada de Normalização (importante para features em escalas diferentes)
    model.add(Normalization())

    # Camada Recorrente LSTM
    model.add(LSTM(
        n_neurons,
        kernel_regularizer=l2(0.001),  # Regularização L2
        activation='relu'  # Ativação ReLU
    ))

    # Camada de Dropout para regularização
    model.add(Dropout(0.4))

    # Camada de Classificação
    model.add(Dense(n_classes, activation='softmax'))

    # Otimizador Adam e Função de Perda
    optimizer = Adam(
        learning_rate=0.0001
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==========================
# 3. Treinamento e Avaliação
# ==========================
def train_model(model, X_train, y_train, X_val, y_val):
    """Treina o modelo com early stopping e retorna o histórico."""
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    )

    print("\nIniciando o treinamento do modelo...")
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        callbacks=[early_stopping]
    )

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo no conjunto de teste."""
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nAcurácia no conjunto de teste: {test_acc:.4f}")


def save_model(model, name="model"):
    """Salva o modelo em dois formatos: H5 e Keras."""
    model.save(f"{name}.h5")
    print("Modelo salvo em formato .h5")
    model.save(f"{name}.keras")
    print("Modelo salvo em formato .keras")
