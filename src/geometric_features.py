import numpy as np
from entities.Settings import GeometricFeaturesSettings, LandmarkSettings


def extract_custom_geometric_features(landmarks_sequence):
    """
    Processa uma sequência de landmarks brutos e extrai o vetor de 88 features.
    """
    return np.array([extract_frame_features(frame) for frame in landmarks_sequence])


def extract_frame_features(frame_landmarks):
    """Extrai as 88 features geométricas de um único frame."""
    all_landmarks = frame_landmarks.reshape(-1, 3)
    pose_landmarks = all_landmarks[LandmarkSettings.POSE_START:LandmarkSettings.POSE_END]
    left_hand_landmarks = all_landmarks[LandmarkSettings.LEFT_HAND_START:LandmarkSettings.LEFT_HAND_END]
    right_hand_landmarks = all_landmarks[LandmarkSettings.RIGHT_HAND_START:LandmarkSettings.RIGHT_HAND_END]

    left_hand_angles = __calculate_hand_angles(left_hand_landmarks)
    right_hand_angles = __calculate_hand_angles(right_hand_landmarks)
    pose_distances = __calculate_pose_distances(pose_landmarks)

    if GeometricFeaturesSettings.USE_LEGACY_FEATURES:
        return np.concatenate([left_hand_angles, right_hand_angles, pose_distances])

    left_hand_distances = __calculate_hand_distances(left_hand_landmarks)
    right_hand_distances = __calculate_hand_distances(right_hand_landmarks)

    left_hand_normal = __calculate_palm_normal(left_hand_landmarks)
    right_hand_normal = __calculate_palm_normal(right_hand_landmarks)

    return np.concatenate([
        left_hand_angles, right_hand_angles,
        pose_distances,
        left_hand_distances, right_hand_distances,
        left_hand_normal, right_hand_normal
    ])


def __compute_angle_between_vectors(v1, v2):
    """Calcula o ângulo entre dois vetores em graus."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def __calculate_hand_angles(hand_landmarks):
    """Calcula 26 ângulos para uma mão."""
    if hand_landmarks.sum() == 0:
        return np.zeros(GeometricFeaturesSettings.NUM_ANGLES_PER_HAND)

    angles = []
    for point1_index, point2_index, point3_index in GeometricFeaturesSettings.HAND_CONNECTIONS_INDEXES:
        point1, point2, point3 = hand_landmarks[point1_index], hand_landmarks[point2_index], hand_landmarks[point3_index]
        angle = __compute_angle_between_vectors(point1 - point2, point3 - point2)
        angles.append(angle)

    # Padding para garantir tamanho fixo
    while len(angles) < GeometricFeaturesSettings.NUM_ANGLES_PER_HAND:
        angles.append(0)

    return np.array(angles[:GeometricFeaturesSettings.NUM_ANGLES_PER_HAND])


def __calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def __calculate_torso_size(pose_landmarks):
    left_shoulder, right_shoulder = pose_landmarks[11], pose_landmarks[12]
    left_hip, right_hip = pose_landmarks[23], pose_landmarks[24]

    pose_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    return __calculate_distance(pose_center, hip_center), pose_center


def __normalize_pose_landmarks(pose_landmarks):
    """Normaliza os landmarks da pose com base no centro do torso."""
    torso_size, pose_center = __calculate_torso_size(pose_landmarks)

    if torso_size < 1e-6:  # Evita divisão por zero
        return None

    return (pose_landmarks - pose_center) / torso_size


def __calculate_pose_distances(pose_landmarks):
    """Normaliza a pose e calcula 38 distâncias."""
    if pose_landmarks.sum() == 0:
        return np.zeros(GeometricFeaturesSettings.NUM_POSE_DISTANCES)

    normalized_landmarks = __normalize_pose_landmarks(pose_landmarks)
    if normalized_landmarks is None:
        return np.zeros(GeometricFeaturesSettings.NUM_POSE_DISTANCES)

    torso_size, _ = __calculate_torso_size(normalized_landmarks)

    distances = [torso_size]
    for point1_index, point2_index in GeometricFeaturesSettings.POSE_PAIRS_INDEXES:
        if point1_index < len(normalized_landmarks) and point2_index < len(normalized_landmarks):
            distance = __calculate_distance(
                normalized_landmarks[point1_index],
                normalized_landmarks[point2_index]
            )
            distances.append(distance)
        else:
            distances.append(0)

    while len(distances) < GeometricFeaturesSettings.NUM_POSE_DISTANCES:
        distances.append(0)

    return np.array(distances[:GeometricFeaturesSettings.NUM_POSE_DISTANCES])


def __calculate_hand_scale(hand_landmarks):
    """
    Calcula a escala da mão baseada na distância entre o Pulso (0) e a base do dedo médio (9).
    Usado para normalizar outras distâncias da mão.
    """
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    return np.linalg.norm(middle_mcp - wrist)


def __calculate_palm_normal(hand_landmarks):
    """
    Calcula o vetor normal da palma da mão.
    Usa o produto vetorial entre (IndexMCP - Wrist) e (PinkyMCP - Wrist).
    Retorna vetor normalizado (x, y, z).
    """
    if hand_landmarks.sum() == 0:
        return np.zeros(3)

    wrist_idx, index_mcp_idx, pinky_mcp_idx = GeometricFeaturesSettings.HAND_PALM_NORMAL_INDICES
    
    wrist = hand_landmarks[wrist_idx]
    index_mcp = hand_landmarks[index_mcp_idx]
    pinky_mcp = hand_landmarks[pinky_mcp_idx]

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm < 1e-6:
        return np.zeros(3)

    return normal / norm


def __calculate_hand_distances(hand_landmarks):
    """
    Calcula distâncias normalizadas entre:
    - Pontas dos dedos e ponta do polegar.
    - Pulso e pontas dos dedos.
    """
    if hand_landmarks.sum() == 0:
        return np.zeros(GeometricFeaturesSettings.NUM_DISTANCES_PER_HAND)

    scale = __calculate_hand_scale(hand_landmarks)
    if scale < 1e-6:
        return np.zeros(GeometricFeaturesSettings.NUM_DISTANCES_PER_HAND)

    distances = []
    
    # Distâncias Dedos-Polegar
    for p1_idx, p2_idx in GeometricFeaturesSettings.HAND_FINGERTIP_THUMB_PAIRS:
        dist = np.linalg.norm(hand_landmarks[p1_idx] - hand_landmarks[p2_idx])
        distances.append(dist / scale)

    # Distâncias Pulso-Pontas
    for p1_idx, p2_idx in GeometricFeaturesSettings.HAND_WRIST_FINGERTIP_PAIRS:
        dist = np.linalg.norm(hand_landmarks[p1_idx] - hand_landmarks[p2_idx])
        distances.append(dist / scale)

    return np.array(distances)

