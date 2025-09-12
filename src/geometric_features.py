import numpy as np


# =============================================================================
# Seção 4.4: Extração de Características Geométricas Customizadas
# =============================================================================

# ==========================
# Utilitários para Ângulos
# ==========================
def get_hand_connections():
    """Retorna a lista de tríades de índices usadas para cálculo de ângulos da mão."""
    return [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),  # Polegar
        (0, 5, 6), (5, 6, 7), (6, 7, 8),  # Indicador
        (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Médio
        (0, 13, 14), (13, 14, 15), (14, 15, 16),  # Anelar
        (0, 17, 18), (17, 18, 19), (18, 19, 20),  # Mínimo
        (5, 9, 13), (9, 13, 17)  # Entre dedos
    ]


def compute_angle_between_vectors(v1, v2):
    """Calcula o ângulo entre dois vetores em graus."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def calculate_hand_angles(hand_landmarks):
    """Calcula 26 ângulos para uma mão. (Seção 4.4.1)"""
    ANGLES_PER_HAND = 26
    if hand_landmarks.sum() == 0:
        return np.zeros(ANGLES_PER_HAND)

    angles = []
    for p1_idx, p2_idx, p3_idx in get_hand_connections():
        p1, p2, p3 = hand_landmarks[p1_idx], hand_landmarks[p2_idx], hand_landmarks[p3_idx]
        angle = compute_angle_between_vectors(p1 - p2, p3 - p2)
        angles.append(angle)

    # Padding para garantir tamanho fixo
    while len(angles) < ANGLES_PER_HAND:
        angles.append(0)

    return np.array(angles[:ANGLES_PER_HAND])


# ==========================
# Utilitários para Pose
# ==========================

def normalize_pose_landmarks(pose_landmarks):
    """Normaliza os landmarks da pose com base no centro do torso."""
    left_shoulder, right_shoulder = pose_landmarks[11], pose_landmarks[12]
    left_hip, right_hip = pose_landmarks[23], pose_landmarks[24]

    pose_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    torso_size = np.linalg.norm(pose_center - hip_center)

    if torso_size < 1e-6:  # Evita divisão por zero
        return None

    return (pose_landmarks - pose_center) / torso_size


def get_pose_pairs():
    """Retorna a lista de pares de índices para cálculo de distâncias da pose."""
    return [
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
        (15, 17), (16, 18), (17, 19), (18, 20), (19, 21),
        (20, 22), (21, 23), (22, 24), (23, 25), (24, 26),
        (11, 23), (12, 24), (13, 25), (14, 26), (15, 27),
        (16, 28), (17, 29), (18, 30), (19, 31), (20, 32),
        (0, 11), (0, 12), (0, 23), (0, 24), (1, 11),
        (2, 12), (3, 23), (4, 24), (5, 11), (6, 12),
        (7, 23), (8, 24), (9, 11), (10, 12)
    ]


def calculate_pose_distances(pose_landmarks):
    """Normaliza a pose e calcula 38 distâncias. (Seção 4.4.2)"""
    NUM_POSE_DISTANCES = 38
    if pose_landmarks.sum() == 0:
        return np.zeros(NUM_POSE_DISTANCES)

    normalized_landmarks = normalize_pose_landmarks(pose_landmarks)
    if normalized_landmarks is None:
        return np.zeros(NUM_POSE_DISTANCES)

    distances = []
    for p1_idx, p2_idx in get_pose_pairs():
        if p1_idx < len(normalized_landmarks) and p2_idx < len(normalized_landmarks):
            distances.append(np.linalg.norm(normalized_landmarks[p1_idx] - normalized_landmarks[p2_idx]))
        else:
            distances.append(0)

    while len(distances) < NUM_POSE_DISTANCES:
        distances.append(0)

    return np.array(distances[:NUM_POSE_DISTANCES])


# ==========================
# Função Principal
# ==========================

def extract_frame_features(frame_landmarks):
    """Extrai as 90 features geométricas de um único frame."""
    all_landmarks = frame_landmarks.reshape(-1, 3)
    pose_landmarks = all_landmarks[0:33]
    left_hand_landmarks = all_landmarks[501:522]
    right_hand_landmarks = all_landmarks[522:543]

    left_hand_angles = calculate_hand_angles(left_hand_landmarks)
    right_hand_angles = calculate_hand_angles(right_hand_landmarks)
    pose_distances = calculate_pose_distances(pose_landmarks)

    return np.concatenate([left_hand_angles, right_hand_angles, pose_distances])


def extract_custom_geometric_features(landmarks_sequence):
    """
    Processa uma sequência de landmarks brutos e extrai o vetor de 90 features.
    """
    return np.array([extract_frame_features(frame) for frame in landmarks_sequence])
