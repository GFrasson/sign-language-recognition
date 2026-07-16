class Settings:
    NUM_FRAMES: int = 30
    LSTM_UNITS: int = 512
    DATA_PATH: str = "data/videos"
    FEATURES_PATH: str = "data/features-hands-distances-normal-face-126-frames-30"
    MODELS_PATH: str = "models"
    SEED: int = 42
    VIDEO_WIDTH: int = 640
    VIDEO_HEIGHT: int = 480

    # Defines which classes constitute a specialist group for a given trigger class
    SPECIALIST_CONFIGS: dict[int, list[int]] = {
        4: [4, 7],
        16: [16, 17]
    }


class LandmarkSettings:
    POSE_COUNT: int = 33
    FACE_COUNT: int = 468
    HAND_COUNT: int = 21

    # Indicies (offsets) in the flattened vector [Pose, Face, LH, RH]
    POSE_START: int = 0
    POSE_END: int = POSE_START + POSE_COUNT

    FACE_START: int = POSE_END
    FACE_END: int = FACE_START + FACE_COUNT

    LEFT_HAND_START: int = FACE_END
    LEFT_HAND_END: int = LEFT_HAND_START + HAND_COUNT

    RIGHT_HAND_START: int = LEFT_HAND_END
    RIGHT_HAND_END: int = RIGHT_HAND_START + HAND_COUNT

    TOTAL_LANDMARKS: int = RIGHT_HAND_END


class ModelSettings:
    BATCH_SIZE: int = 1024
    EPOCHS: int = 200
    LEARNING_RATE: float = 0.0001
    WEIGHT_DECAY: float = 0.005
    DROPOUT_RATE: float = 0.4
    EARLY_STOPPING_PATIENCE: int = 20
    SPECIALIST_LSTM_UNITS: int = 512
    SPECIALIST_LSTM_UNITS_2: int = 64  # Second LSTM layer for deeper architecture
    SPECIALIST_BATCH_SIZE: int = 32
    SPECIALIST_EARLY_STOPPING_PATIENCE: int = 20
    SPECIALIST_DROPOUT_RATE: float = 0.4
    SPECIALIST_WEIGHT_DECAY: float = 0.005
    SPECIALIST_LEARNING_RATE: float = 0.0001
    LSTM_UNROLL: bool = False


class GeometricFeaturesSettings:
    HAND_CONNECTIONS_INDEXES: list[tuple[int, int, int]] = [
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

    # --- Configuration ---
    USE_LEGACY_FEATURES: bool = False

    POSE_PAIRS_INDEXES_LEGACY: list[tuple[int, int]] = [
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

    POSE_PAIRS_INDEXES_NEW: list[tuple[int, int]] = [
        (0, 15), (0, 16),  # Nariz e pulsos
        (12, 16), (12, 15), (11, 16), (11, 15),  # Ombros e pulsos
        (12, 14), (12, 13), (11, 14), (11, 13),  # Ombros e cotovelos
        (16, 17), (15, 18),  # Pulsos cruzados com dedinhos (Esq-Dir, Dir-Esq)
        (16, 19), (15, 20),  # Pulsos cruzados com indicadores
        (18, 19), (17, 20),  # Dedinhos cruzados com indicadores
        (18, 21), (17, 22),  # Dedinhos cruzados com polegares
        (20, 21), (19, 22),  # Indicadores cruzados com polegares
        (21, 22),  # Polegares esquerdo e direito
        (19, 20),  # Indicadores esquerdo e direito
        (17, 18),  # Mindinhos esquerdo e direito
        (15, 16),  # Pulsos esquerdo e direito
        (13, 14)  # Cotovelos esquerdo e direito
    ]

    # Default to NEW list
    POSE_PAIRS_INDEXES: list[tuple[int, int]] = POSE_PAIRS_INDEXES_NEW

    HAND_PALM_NORMAL_INDICES: list[int] = [0, 5, 17]  # Wrist, IndexMCP, PinkyMCP

    HAND_FINGERTIP_THUMB_PAIRS: list[tuple[int, int]] = [
        (4, 8), (4, 12), (4, 16), (4, 20)  # Thumb tip to other tips
    ]

    HAND_WRIST_FINGERTIP_PAIRS: list[tuple[int, int]] = [
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20)  # Wrist to all tips
    ]

    FACE_ANCHOR_INDICES: list[int] = [
        33,   # Left Eye (Inner)
        263,  # Right Eye (Inner)
        234,  # Left Ear
        454,  # Right Ear
        61,   # Mouth Left
        291   # Mouth Right
    ]

    NUM_ANGLES_PER_HAND: int = len(HAND_CONNECTIONS_INDEXES)
    NUM_DISTANCES_PER_HAND: int = len(HAND_FINGERTIP_THUMB_PAIRS) + len(HAND_WRIST_FINGERTIP_PAIRS)
    NUM_FACE_ANCHORS: int = len(FACE_ANCHOR_INDICES)
    
    # +1 para a distância do torso
    NUM_POSE_DISTANCES: int = len(POSE_PAIRS_INDEXES) + 1
    
    # Features (calculated dynamically in configure(), but set defaults here)
    N_FEATURES: int = (2 * NUM_ANGLES_PER_HAND) + NUM_POSE_DISTANCES + (2 * NUM_DISTANCES_PER_HAND) + (2 * 3) + (4 * NUM_FACE_ANCHORS)
    
    # Velocity features configuration
    USE_VELOCITY_FEATURES: bool = False
    
    # Detailed feature counts (populated in configure)
    NUM_GEOMETRIC_FEATURES: int = 0
    NUM_VELOCITY_FEATURES: int = 0

    @classmethod
    def configure(cls, use_legacy: bool, use_velocity: bool = False):
        cls.USE_LEGACY_FEATURES = use_legacy
        cls.USE_VELOCITY_FEATURES = use_velocity
        
        if use_legacy:
            cls.POSE_PAIRS_INDEXES = cls.POSE_PAIRS_INDEXES_LEGACY
            cls.NUM_POSE_DISTANCES = len(cls.POSE_PAIRS_INDEXES) + 1
            # Legacy: Angles (52) + Pose (36) = 88
            base_features = (2 * cls.NUM_ANGLES_PER_HAND) + cls.NUM_POSE_DISTANCES
        else:
            cls.POSE_PAIRS_INDEXES = cls.POSE_PAIRS_INDEXES_NEW
            cls.NUM_POSE_DISTANCES = len(cls.POSE_PAIRS_INDEXES) + 1
            # New: Angles (52) + Pose (26) + Distances (18) + Normal (6) + FaceDist (24) = 126
            base_features = (2 * cls.NUM_ANGLES_PER_HAND) + cls.NUM_POSE_DISTANCES + (2 * cls.NUM_DISTANCES_PER_HAND) + (2 * 3) + (4 * cls.NUM_FACE_ANCHORS)
        
        cls.NUM_GEOMETRIC_FEATURES = base_features
        
        if use_velocity:
            cls.NUM_VELOCITY_FEATURES = VelocityFeaturesSettings.N_VELOCITY_FEATURES
            cls.N_FEATURES = base_features + VelocityFeaturesSettings.N_VELOCITY_FEATURES
        else:
            cls.NUM_VELOCITY_FEATURES = 0
            cls.N_FEATURES = base_features

class VelocityFeaturesSettings:
    """Configuration for velocity feature extraction."""

    # Key landmarks to track for velocity (indices within each hand)
    # Wrist, Thumb tip, Index tip, Middle tip, Ring tip, Pinky tip
    HAND_KEY_POINTS: list[int] = [0, 4, 8, 12, 16, 20]

    # Pose key points for body movement tracking
    # Nose, Left wrist, Right wrist
    POSE_KEY_POINTS: list[int] = [0, 15, 16]

    # Number of features per hand: 6 points * 4 features (vel_mag, acc_mag, dir_x, dir_y)
    NUM_FEATURES_PER_HAND: int = len(HAND_KEY_POINTS) * 4

    # Number of features for pose: 3 points * 4 features
    NUM_POSE_FEATURES: int = len(POSE_KEY_POINTS) * 4

    # Global kinematic features: peak velocity frame, velocity variance, etc.
    NUM_KINEMATIC_FEATURES: int = 6

    # Total features per frame
    N_VELOCITY_FEATURES: int = (2 * NUM_FEATURES_PER_HAND) + NUM_POSE_FEATURES + NUM_KINEMATIC_FEATURES
