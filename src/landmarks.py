import random

import numpy as np
import mediapipe as mp
from entities.Settings import Settings
from entities.Video import Video
from entities.VideoFrame import VideoFrame


def read_all_video_frames(video_path: str) -> list[VideoFrame]:
    """Lê TODOS os frames do vídeo."""
    try:
        video = Video(video_path)
    except ValueError as e:
        print(f"Error opening video {video_path}: {e}")
        return None

    video_frames = []
    # Read all frames regardless of count
    # Video class might not support iteration? Let's check. 
    # VideoFrame usually has total_frames.
    
    # We can iterate until None is returned or use total_frames.
    # Assuming standard OpenCV behavior in Video class wrapper.
    # Based on previous code, video.read_frame(index) works.
    
    # Optimally, we process sequentially for speed rather than seeking.
    # But read_frame likely does seeking if index is passed?
    # Let's inspect Video class indirectly or assume sequential read is best.
    # If Video class supports seekless read, that's better. 
    # But previous code used indices.
    
    # Let's try reading all frames using indices 0 to total_frames-1
    for i in range(video.total_frames):
        vf = video.read_frame(i)
        if vf is not None:
            video_frames.append(vf)
            
    video.release()
    
    if len(video_frames) == 0:
        return None
        
    return video_frames


def sample_frames_from_list(all_frames: list[VideoFrame], num_frames: int = Settings.NUM_FRAMES) -> list[VideoFrame]:
    """Seleciona frames aleatórios de uma lista já carregada (distribuição normal)."""
    total_frames = len(all_frames)
    
    if total_frames < num_frames:
        # If fewer frames than needed, duplicate or take all?
        # Standard behavior: take all and maybe pad? 
        # But previous code returned None if total < num.
        # Let's align with previous behavior -> Return None or maybe duplicate?
        # Reverting to 'None' for consistency with landmarks.py:25
        return None

    # Use same logic as __sample_frame_indices but on range 0..len(all_frames)
    frame_indices = __sample_frame_indices(total_frames, num_frames)
    
    selected_frames = [all_frames[i] for i in frame_indices]
    return selected_frames


def read_video_frames(video_path: str, num_frames: int = Settings.NUM_FRAMES) -> list[VideoFrame]:
    """Lê frames do vídeo selecionados via distribuição normal."""
    # ... existing implementation ...
    try:
        video = Video(video_path)
    except ValueError as e:
        print(f"Error opening video {video_path}: {e}")
        return None

    total_frames = video.total_frames
    if total_frames < num_frames:
        video.release()
        return None

    frame_indices = __sample_frame_indices(total_frames, num_frames)
    
    video_frames = []
    for frame_idx in frame_indices:
        vf = video.read_frame(frame_idx)
        if vf is not None:
            video_frames.append(vf)
            
    video.release()
    
    if len(video_frames) < num_frames:
        return None
        
    return video_frames


def process_frames(video_frames: list[VideoFrame], augment: bool = False, augment_params = None) -> np.ndarray:
    """Processa uma lista de frames (VideoFrame) e retorna landmarks."""
    
    # Apply augmentation if needed (on a copy/in-place? VideoFrame methods mutate?)
    # VideoFrame methods usually mutate self or return self? 
    # Let's assume we need to clone if we want to preserve original for other augs.
    # But here we receive a list of frames that we are allowed to modify or processed frames.
    
    frames_to_process = []
    if augment:
        # Generate params if not provided, though typically passed from outside for consistency? 
        # Current logic generates random params here if augment=True.
        if augment_params is None:
            augment_params = __random_augmentation_params()
            
        for vf in video_frames:
            # We MUST clone the frame because specific augmentation modifies it in place
            # and we want to reuse the original frames for other augmentations.
            # VideoFrame is likely a wrapper around a numpy array. 
            # We need deep copy of the image.
            
            # Assuming VideoFrame needs a copy method or we reconstruct it.
            # Looking at codebase, VideoFrame seems to hold .frame (numpy)
            # Let's add a safe copy mechanism here if VideoFrame doesn't support it, 
            # or rely on the caller to pass fresh copies. 
            # For safety with current VideoFrame implementation (unknown to me in detail), 
            # let's assume methods modify in place.
            
            # Since we don't have a visible clone method, we rely on the fact that
            # __augment_frame modifies the object. 
            # We should create a NEW VideoFrame object with a copy of the numpy array.
            
            new_vf = VideoFrame(vf.frame.copy())
            frames_to_process.append(__augment_frame(new_vf, augment_params))
    else:
        # If not augmenting, we can just use the frames. 
        # But wait, resize() in __process_frame modifies in place too?
        # "__process_frame: video_frame.resize(...)"
        # Yes, it likely modifies. So we ALWAYS need copies if we want to reuse the source frames.
        frames_to_process = [VideoFrame(vf.frame.copy()) for vf in video_frames]

    # Create models once and reuse them
    holistic_model = create_holistic_model()

    try:
        landmarks_sequence = [__process_frame(vf, holistic_model) for vf in frames_to_process]
    finally:
        holistic_model.close()
        # hands_model.close()
    
    return np.array(landmarks_sequence)


def extract_landmarks(video_path: str, num_frames: int = Settings.NUM_FRAMES, augment: bool = False):
    """
    Mantém compatibilidade com código legado, mas usa as novas funções.
    """
    frames = read_video_frames(video_path, num_frames)
    if frames is None:
        return None
        
    return process_frames(frames, augment)


def create_holistic_model():
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


def __process_frame(video_frame: VideoFrame, holistic_model):
    """Processa um frame, extrai e concatena todos os landmarks em um único vetor."""
    video_frame.resize(Settings.VIDEO_WIDTH, Settings.VIDEO_HEIGHT).bgr_to_rgb()
    
    results = holistic_model.process(video_frame.frame)

    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])


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
