import numpy as np


# =============================================================================
# Seção 4.4: Extração de Características Geométricas Customizadas
# =============================================================================

# ==========================
# Utilitários para Ângulos
# ==========================
HAND_CONNECTIONS_INDEXES = [
    (4, 3, 2), (3, 2, 1),  # Polegar
    (8, 7, 6), (7, 6, 5),  # Indicador
    (12, 11, 10), (11, 10, 9),  # Médio
    (16, 15, 14), (15, 14, 13),  # Anelar
    (20, 19, 18), (19, 18, 17),  # Mínimo
    (2, 1, 0), (1, 0, 5), (1, 0, 17), (5, 0, 17),  # Palma
    (18, 17, 0), (6, 5, 0),  # Ligações dedos / palma
    (6, 5, 9), (5, 9, 10), (10, 9, 13), (9, 13, 14), (14, 13, 17), (13, 17, 18),  # Ligações entre dedos
    (5, 9, 13), (9, 13, 17),  # Palma (entre dedos)
    (0, 17, 13), (0, 5, 9)  # Palma (base dos dedos)
]

POSE_PAIRS_INDEXES = [
    (0, 15), (0, 16),  # Nariz e pulsos
    (12, 16), (12, 15), (11, 16), (11, 15),  # Ombros e pulsos
    (12, 14), (12, 13), (11, 14), (11, 13),  # Ombros e cotovelos
    (16, 18), (16, 17), (15, 17), (15, 18),  # Pulsos e dedos mindinhos
    (16, 20), (16, 19), (15, 20), (15, 19),  # Pulsos e dedos indicadores
    (18, 20), (18, 19), (17, 20), (17, 19),  # Dedos mindinhos e indicadores
    (18, 22), (18, 21), (17, 21), (17, 22),  # Dedos mindinhos e polegares
    (20, 22), (20, 21), (19, 21), (19, 22),  # Dedos indicadores e polegares
    (21, 22),  # Polegares esquerdo e direito
    (19, 20),  # Indicadores esquerdo e direito
    (17, 18),  # Mindinhos esquerdo e direito
    (15, 16),  # Pulsos esquerdo e direito
    (13, 14)  # Cotovelos esquerdo e direito
]

NUM_ANGLES_PER_HAND = len(HAND_CONNECTIONS_INDEXES)
NUM_POSE_DISTANCES = len(POSE_PAIRS_INDEXES) + 1  # +1 para a distância do torso


def compute_angle_between_vectors(v1, v2):
    """Calcula o ângulo entre dois vetores em graus."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def calculate_hand_angles(hand_landmarks):
    """Calcula 26 ângulos para uma mão. (Seção 4.4.1)"""
    if hand_landmarks.sum() == 0:
        return np.zeros(NUM_ANGLES_PER_HAND)

    angles = []
    for point1_index, point2_index, point3_index in HAND_CONNECTIONS_INDEXES:
        point1, point2, point3 = hand_landmarks[point1_index], hand_landmarks[
            point2_index], hand_landmarks[point3_index]
        angle = compute_angle_between_vectors(point1 - point2, point3 - point2)
        angles.append(angle)

    # Padding para garantir tamanho fixo
    while len(angles) < NUM_ANGLES_PER_HAND:
        angles.append(0)

    return np.array(angles[:NUM_ANGLES_PER_HAND])


# ==========================
# Utilitários para Pose
# ==========================
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def calculate_torso_size(pose_landmarks):
    left_shoulder, right_shoulder = pose_landmarks[11], pose_landmarks[12]
    left_hip, right_hip = pose_landmarks[23], pose_landmarks[24]

    pose_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    return calculate_distance(pose_center, hip_center), pose_center


def normalize_pose_landmarks(pose_landmarks):
    """Normaliza os landmarks da pose com base no centro do torso."""
    torso_size, pose_center = calculate_torso_size(pose_landmarks)

    if torso_size < 1e-6:  # Evita divisão por zero
        return None

    return (pose_landmarks - pose_center) / torso_size


def calculate_pose_distances(pose_landmarks):
    """Normaliza a pose e calcula 38 distâncias. (Seção 4.4.2)"""
    if pose_landmarks.sum() == 0:
        return np.zeros(NUM_POSE_DISTANCES)

    normalized_landmarks = normalize_pose_landmarks(pose_landmarks)
    if normalized_landmarks is None:
        return np.zeros(NUM_POSE_DISTANCES)

    torso_size, _ = calculate_torso_size(normalized_landmarks)

    distances = [torso_size]
    for point1_index, point2_index in POSE_PAIRS_INDEXES:
        if point1_index < len(normalized_landmarks) and point2_index < len(normalized_landmarks):
            distance = calculate_distance(
                normalized_landmarks[point1_index],
                normalized_landmarks[point2_index]
            )
            distances.append(distance)
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
