import cv2
import numpy as np
import mediapipe as mp
import random
import concurrent.futures


# =============================================================================
# Seção 4.3: Estimação de Landmarks com MediaPipe
# =============================================================================
class LandmarkExtractor:
    def __init__(self):
        pass

    def create_holistic_model(self):
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


    def open_video(self, video_path):
        """Abre o vídeo e retorna o objeto VideoCapture ou None em caso de erro."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print(f"Erro ao abrir o vídeo: {video_path}")
            return None
        return capture


    def sample_frame_indices(self, total_frames, num_frames):
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


    def process_frame(self, frame):
        """Processa um frame, extrai e concatena todos os landmarks em um único vetor."""
        holistic_model = self.create_holistic_model()
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(frame_rgb)
        holistic_model.close()

        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 3)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
            if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)

        return np.concatenate([pose, face, lh, rh])


    def augment_frame(self, frame, params):
        """Aplica as técnicas de aumento de dados em um frame."""
        # Espelhamento horizontal
        if params['flip']:
            frame = cv2.flip(frame, 1)

        # Rotação
        if params['rotation'] != 0:
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), params['rotation'], 1)
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Translação
        if params['tx'] != 0 or params['ty'] != 0:
            M = np.float32([
                [1, 0, params['tx']],
                [0, 1, params['ty']]
            ])
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)

        # Corte centralizado
        if params['crop'] > 0:
            h, w = frame.shape[:2]
            crop_h = int(h * params['crop'])
            crop_w = int(w * params['crop'])
            frame = frame[crop_h:h - crop_h, crop_w:w - crop_w]
            frame = cv2.resize(frame, (w, h))

        # Alteração de brilho
        if params['brightness'] != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=params['brightness'], beta=0)

        # Alteração de contraste
        if params['contrast'] != 0:
            frame = cv2.add(frame, np.array([params['contrast']]))

        return frame


    def random_augmentation_params(self):
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


    def extract_landmarks(self, video_path, num_frames=15, augment=False):
        """
        Extrai landmarks de um vídeo usando o MediaPipe Holistic.
        Se augment=True, aplica aumento de dados com parâmetros fixos para o vídeo.
        Agora processa frames em paralelo.
        """
        video_capture = self.open_video(video_path)
        if video_capture is None:
            return None

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            video_capture.release()
            return None

        frame_indices = self.sample_frame_indices(total_frames, num_frames)
        params = self.random_augmentation_params() if augment else None

        frames = []
        for frame_idx in frame_indices:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            if success:
                if augment:
                    frame = self.augment_frame(frame, params)
                frames.append(frame)
        video_capture.release()

        # Parallel processing of frames
        with concurrent.futures.ThreadPoolExecutor() as executor:
            landmarks_sequence = list(executor.map(self.process_frame, frames))

        return np.array(landmarks_sequence)
