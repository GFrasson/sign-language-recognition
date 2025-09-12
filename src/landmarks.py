import cv2
import numpy as np
import mediapipe as mp


# =============================================================================
# Seção 4.3: Estimação de Landmarks com MediaPipe
# =============================================================================
def create_holistic_model():
    """Cria e retorna o modelo Holistic do MediaPipe com as configurações da Tabela 3."""
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    )


def open_video(video_path):
    """Abre o vídeo e retorna o objeto VideoCapture ou None em caso de erro."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return None
    return capture


def sample_frame_indices(total_frames, num_frames):
    """
    Seleciona índices de frames usando amostragem Normal (Seção 4.2.1).
    Garante exatamente `num_frames` índices únicos.
    """
    mean = total_frames / 2
    std_dev = mean * 0.4
    frame_indices = np.random.normal(mean, std_dev, num_frames)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1).astype(int)
    frame_indices = sorted(np.unique(frame_indices))

    while len(frame_indices) < num_frames:
        extra_frame = np.random.randint(0, total_frames)
        if extra_frame not in frame_indices:
            frame_indices = sorted(np.append(frame_indices, extra_frame))

    return frame_indices


def process_frame(frame, holistic_model):
    """Processa um frame, extrai e concatena todos os landmarks em um único vetor."""
    frame_resized = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(frame_rgb)

    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])


def extract_landmarks(video_path, num_frames=15):
    """
    Extrai landmarks de um vídeo usando o MediaPipe Holistic.
    O vídeo é redimensionado para 640x480 para consistência (Seção 4.2.3).
    """
    holistic_model = create_holistic_model()
    video_capture = open_video(video_path)
    if video_capture is None:
        return None

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        video_capture.release()
        return None

    frame_indices = sample_frame_indices(total_frames, num_frames)

    landmarks_sequence = []
    for frame_idx in frame_indices:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video_capture.read()
        if success:
            landmarks = process_frame(frame, holistic_model)
            landmarks_sequence.append(landmarks)

    video_capture.release()
    return np.array(landmarks_sequence)
