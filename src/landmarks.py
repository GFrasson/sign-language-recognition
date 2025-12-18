import random

import numpy as np
import mediapipe as mp
from entities.Settings import Settings
from entities.Video import Video
from entities.VideoFrame import VideoFrame


def extract_landmarks(video_path: str, num_frames: int = Settings.NUM_FRAMES, augment: bool = False):
    """
    Extrai landmarks de um vídeo usando o MediaPipe Holistic.
    Se augment=True, aplica aumento de dados com parâmetros fixos para o vídeo.
    Agora processa frames em paralelo.
    """
    video: Video = None

    try:
        video = Video(video_path)
    except ValueError as e:
        print(e)
        return None

    total_frames = video.total_frames
    if total_frames < num_frames:
        video.release()
        return None

    frame_indices = __sample_frame_indices(total_frames, num_frames)
    params = __random_augmentation_params() if augment else None

    video_frames: list[VideoFrame] = []
    for frame_idx in frame_indices:
        video_frame = video.read_frame(frame_idx)
        
        if video_frame is not None:
            if augment:
                video_frame = __augment_frame(video_frame, params)
            video_frames.append(video_frame)
    
    video.release()

    # Sequential processing of frames to save memory
    # Create models once and reuse them
    pose_model = create_pose_model()
    hands_model = create_hands_model()

    try:
        landmarks_sequence = [__process_frame(vf, pose_model, hands_model) for vf in video_frames]
    finally:
        pose_model.close()
        hands_model.close()
    
    return np.array(landmarks_sequence)


def create_pose_model():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    )


def create_hands_model():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    )


def __sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    """
    Seleciona índices de frames usando amostragem Normal.
    Garante exatamente `num_frames` índices únicos.
    """
    frame_indices = __normal_distribution_sample(total_frames, num_frames)

    while len(frame_indices) < num_frames:
        extra_num_frames = num_frames - len(frame_indices)
        extra_indices = __normal_distribution_sample(total_frames, extra_num_frames)

        for index in extra_indices:
            if index not in frame_indices:
                frame_indices = np.append(frame_indices, index)
            if len(frame_indices) == num_frames:
                break

    return sorted(frame_indices)


def __normal_distribution_sample(total_frames: int, num_frames: int) -> list[int]:
    """Gera amostras de índices de frames usando distribuição Normal."""
    mean = total_frames / 2
    std_dev = mean * 0.4

    frame_indices = np.random.normal(mean, std_dev, num_frames)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1).astype(int)
    return sorted(np.unique(frame_indices))


def __process_frame(video_frame: VideoFrame, pose_model, hands_model):
    """Processa um frame, extrai e concatena todos os landmarks em um único vetor."""
    video_frame.resize(Settings.VIDEO_WIDTH, Settings.VIDEO_HEIGHT).bgr_to_rgb()
    
    # Process Pose
    pose_results = pose_model.process(video_frame.frame)
    pose = np.array([[res.x, res.y, res.z] for res in pose_results.pose_landmarks.landmark]).flatten() \
        if pose_results.pose_landmarks else np.zeros(33 * 3)

    # Process Hands
    hands_results = hands_model.process(video_frame.frame)
    
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if hands_results.multi_hand_landmarks:
        for idx, hand_handedness in enumerate(hands_results.multi_handedness):
            label = hand_handedness.classification[0].label
            landmarks = hands_results.multi_hand_landmarks[idx]
            
            flat_landmarks = np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
            
            if label == 'Left':
                lh = flat_landmarks
            elif label == 'Right':
                rh = flat_landmarks

    return np.concatenate([pose, lh, rh])


def __augment_frame(video_frame: VideoFrame, params) -> VideoFrame:
    """Aplica as técnicas de aumento de dados em um frame."""
    # Espelhamento horizontal
    if params['flip']:
        video_frame.flip(1)

    # Rotação
    if params['rotation'] != 0:
        video_frame.rotate(params['rotation'])

    # Translação
    if params['tx'] != 0 or params['ty'] != 0:
        video_frame.translate(params['tx'], params['ty'])
    
    # Corte centralizado
    if params['crop'] > 0:
        video_frame.crop_centralize(params['crop'])

    # Alteração de brilho
    if params['brightness'] != 1.0:
        video_frame.change_brightness(params['brightness'])

    # Alteração de contraste
    if params['contrast'] != 0:
        video_frame.change_contrast(params['contrast'])

    return video_frame


def __random_augmentation_params():
    """Gera parâmetros aleatórios para aumento de dados."""
    return {
        'flip': random.choice([True, False]),
        'rotation': random.uniform(-5, 5),
        'tx': random.randint(-20, 20),
        'ty': random.randint(-20, 20),
        'crop': random.uniform(0, 0.1),
        'brightness': random.uniform(0.7, 1.3),
        'contrast': random.randint(-20, 20)
    }
