import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Masking, Normalization, LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

# =============================================================================
# Seção 4.3: Estimação de Landmarks com MediaPipe
# =============================================================================
# Configuração exata do MediaPipe Holistic conforme Tabela 3 da dissertação.
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)


def extract_landmarks(video_path, num_frames=15):
    """
    Extrai landmarks de um vídeo usando o MediaPipe Holistic.
    O vídeo é primeiro redimensionado para 640x480 para consistência,
    conforme Seção 4.2.3.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return None

    # Lógica de amostragem de frames (aqui simplificada, mas deve corresponder
    # à amostragem Normal ou Uniforme da Seção 4.2.1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        return None

    # Amostragem Normal (Seção 4.2.1)
    mean = total_frames / 2
    std_dev = mean * 0.4
    frame_indices = np.random.normal(mean, std_dev, num_frames)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1).astype(int)
    frame_indices = sorted(np.unique(frame_indices))

    # Garante que tenhamos exatamente num_frames, preenchendo se necessário
    while len(frame_indices) < num_frames:
        extra_frame = np.random.randint(0, total_frames)
        if extra_frame not in frame_indices:
            frame_indices = sorted(np.append(frame_indices, extra_frame))

    landmarks_sequence = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, image = cap.read()
        if not success:
            continue

        # Redimensionamento para 640x480 (Seção 4.2.3)
        image = cv2.resize(image, (640, 480))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(image_rgb)

        # Concatena todos os 543 landmarks (33 pose, 468 face, 21x2 mãos)
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33*3)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
        ) if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)

        landmarks_sequence.append(np.concatenate([pose, face, lh, rh]))

    cap.release()
    return np.array(landmarks_sequence)

# =============================================================================
# Seção 4.4: Extração de Características Geométricas Customizadas
# =============================================================================
# Esta é a abordagem mais eficaz, resultando em um vetor de 90 dimensões.


def calculate_hand_angles(hand_landmarks):
    """Calcula 26 ângulos para uma mão. (Seção 4.4.1)"""
    if hand_landmarks.sum() == 0:
        return np.zeros(26)

    # Definir conexões da mão (baseado na topologia do MediaPipe)
    connections = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),  # Polegar
        (0, 5, 6), (5, 6, 7), (6, 7, 8),  # Indicador
        (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Médio
        (0, 13, 14), (13, 14, 15), (14, 15, 16),  # Anelar
        (0, 17, 18), (17, 18, 19), (18, 19, 20),  # Mínimo
        (5, 9, 13), (9, 13, 17)  # Entre dedos
    ]

    angles = []
    for p1_idx, p2_idx, p3_idx in connections:
        p1 = hand_landmarks[p1_idx]
        p2 = hand_landmarks[p2_idx]
        p3 = hand_landmarks[p3_idx]

        v1 = p1 - p2
        v2 = p3 - p2

        # Normaliza os vetores para evitar problemas numéricos
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)

        dot_product = np.dot(v1_u, v2_u)
        # Garante que o valor esteja no intervalo [-1, 1] para arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)

        angle = np.arccos(dot_product)
        angles.append(np.degrees(angle))

    # Adiciona ângulos adicionais se necessário para chegar a 26
    while len(angles) < 26:
        angles.append(0)  # Padding

    return np.array(angles[:26])


def calculate_pose_distances(pose_landmarks):
    """Normaliza a pose e calcula 38 distâncias. (Seção 4.4.2)"""
    if pose_landmarks.sum() == 0:
        return np.zeros(38)

    # Pontos de referência da pose (índices do MediaPipe Pose)
    left_shoulder = pose_landmarks[11]  # Índice 11: left_shoulder
    right_shoulder = pose_landmarks[12]  # Índice 12: right_shoulder
    left_hip = pose_landmarks[23]  # Índice 23: left_hip
    right_hip = pose_landmarks[24]  # Índice 24: right_hip

    # Normalização da pose
    pose_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    torso_size = np.linalg.norm(pose_center - hip_center)

    if torso_size < 1e-6:  # Evita divisão por zero
        return np.zeros(38)

    normalized_landmarks = (pose_landmarks - pose_center) / torso_size

    # Pares de landmarks para cálculo de distância (lista da Seção 4.4.2)
    # Selecionando pares importantes para reconhecimento de sinais
    pairs = [
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Ombros e braços
        (15, 17), (16, 18), (17, 19), (18, 20), (19, 21),  # Braços e antebraços
        (20, 22), (21, 23), (22, 24), (23, 25), (24, 26),  # Antebracos e mãos
        (11, 23), (12, 24), (13, 25), (14, 26), (15, 27),  # Distâncias cruzadas
        (16, 28), (17, 29), (18, 30), (19, 31), (20, 32),  # Mais distâncias
        (0, 11), (0, 12), (0, 23), (0, 24), (1, 11),       # Nariz e orelhas
        (2, 12), (3, 23), (4, 24), (5, 11), (6, 12),       # Olhos e orelhas
        (7, 23), (8, 24), (9, 11), (10, 12)                # Mais conexões
    ]

    distances = []
    for p1_idx, p2_idx in pairs:
        if p1_idx < len(normalized_landmarks) and p2_idx < len(normalized_landmarks):
            dist = np.linalg.norm(
                normalized_landmarks[p1_idx] - normalized_landmarks[p2_idx])
            distances.append(dist)
        else:
            distances.append(0)  # Padding para índices inválidos

    # Adiciona distâncias adicionais se necessário para chegar a 38
    while len(distances) < 38:
        distances.append(0)  # Padding

    return np.array(distances[:38])


def extract_custom_geometric_features(landmarks_sequence):
    """
    Processa uma sequência de landmarks brutos e extrai o vetor de 90 features.
    """
    feature_sequence = []
    for frame_landmarks in landmarks_sequence:
        # Reshape do vetor achatado para acessar os landmarks individuais
        all_landmarks = frame_landmarks.reshape(-1, 3)

        pose_landmarks = all_landmarks[0:33]
        # Índices para mão esquerda
        left_hand_landmarks = all_landmarks[501:522]
        # Índices para mão direita
        right_hand_landmarks = all_landmarks[522:543]

        # 4.4.1: Extração de Ângulos das Mãos (26 por mão = 52)
        left_hand_angles = calculate_hand_angles(left_hand_landmarks)
        right_hand_angles = calculate_hand_angles(right_hand_landmarks)

        # 4.4.2: Extração de Distâncias da Pose (38)
        pose_distances = calculate_pose_distances(pose_landmarks)

        # Concatena para formar o vetor final de 90 dimensões
        final_features = np.concatenate(
            [left_hand_angles, right_hand_angles, pose_distances])
        feature_sequence.append(final_features)

    return np.array(feature_sequence)

# =============================================================================
# Seção 4.6: Arquitetura do Modelo Sequencial LSTM
# =============================================================================


def build_lstm_model(n_features, n_neurons, n_classes):
    """
    Constrói o modelo LSTM conforme o Quadro 3 e hiperparâmetros da Seção 4.6.
    """
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
        kernel_regularizer=tf.keras.regularizers.l2(0.001),  # Regularização L2
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

# =============================================================================
# Pipeline Principal e Treinamento
# =============================================================================


def main():
    # --- Parâmetros do Experimento (melhor configuração da dissertação) ---
    NUM_FRAMES = 15
    N_FEATURES = 90  # 52 (ângulos das mãos) + 38 (distâncias da pose)
    LSTM_UNITS = 512  # Melhor resultado conforme Tabela 11
    DATA_PATH = "data/videos"

    # --- 1. Carregamento e Extração de Características ---
    # Mapeamento de classes baseado nos nomes das pastas
    class_mapping = {
        '01-professor': 0,
        '02-estudar': 1,
        '03-aluno': 2,
        '04-duvida': 3,
        '05-perguntar': 4,
        '06-responder': 5,
        '07-entender': 6
    }
    
    NUM_CLASSES = len(class_mapping)
    
    # Buscar todos os vídeos MP4 nas subpastas
    video_files = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    print(f"Encontrados {len(video_files)} vídeos para processar")
    
    X = []
    y = []
    
    for video_file in tqdm(video_files, desc="Extraindo Features"):
        # Extrair o nome da pasta pai (classe)
        folder_name = os.path.basename(os.path.dirname(video_file))
        
        # Verificar se a classe está no mapeamento
        if folder_name in class_mapping:
            label = class_mapping[folder_name]
            
            print(f"Processando: {video_file} -> Classe: {folder_name} (ID: {label})")
            
            raw_landmarks = extract_landmarks(video_file, NUM_FRAMES)
            if raw_landmarks is not None and raw_landmarks.shape[0] == NUM_FRAMES:
                features = extract_custom_geometric_features(raw_landmarks)
                X.append(features)
                y.append(label)
            else:
                print(f"Erro ao processar vídeo: {video_file}")
        else:
            print(f"Classe não encontrada no mapeamento: {folder_name}")
    
    print(f"Total de vídeos processados com sucesso: {len(X)}")
    
    if len(X) == 0:
        print("Nenhum vídeo foi processado com sucesso. Verifique os caminhos e formatos dos arquivos.")
        return
    
    # Converter para arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    print(f"Shape dos dados X: {X.shape}")
    print(f"Shape dos dados y: {y.shape}")
    print(f"Classes únicas encontradas: {np.unique(y)}")
    
    # Divisão dos dados (80% treino, 10% validação, 10% teste)
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.9)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:val_size]
    y_val = y[train_size:val_size]
    X_test = X[val_size:]
    y_test = y[val_size:]
    
    print(f"Dados de treino: {X_train.shape[0]} amostras")
    print(f"Dados de validação: {X_val.shape[0]} amostras")
    print(f"Dados de teste: {X_test.shape[0]} amostras")

    # --- 2. Construção do Modelo ---
    model = build_lstm_model(N_FEATURES, LSTM_UNITS, NUM_CLASSES)
    model.summary()

    # --- 3. Treinamento do Modelo ---
    # Callback de EarlyStopping (Seção 4.6)
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    )

    print("\nIniciando o treinamento do modelo...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,  # Máximo de épocas
        callbacks=[early_stopping]
    )

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


    print("\nTreinamento concluído.")

    # --- 4. Avaliação (exemplo) ---
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nAcurácia no conjunto de teste: {test_acc:.4f}")

    # --- 5. Salvar o Modelo ---
    model.save("model.h5")
    print("Modelo salvo com sucesso.")
    model.save("model.keras")
    print("Modelo salvo com sucesso.")

    # --- 6. Carregar o Modelo ---
    # model = tf.keras.models.load_model("model.h5")
    # print("Modelo carregado com sucesso.")
    # model = tf.keras.models.load_model("model.keras")
    # print("Modelo carregado com sucesso.")
    

if __name__ == '__main__':
    main()
