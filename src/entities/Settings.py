class Settings:
    NUM_FRAMES: int = 15
    LSTM_UNITS: int = 512
    DATA_PATH: str = "data/videos"
    FEATURES_PATH: str = "data/features"
    MODELS_PATH: str = "models"
    SEED: int = 42
    VIDEO_WIDTH: int = 640
    VIDEO_HEIGHT: int = 480


class ModelSettings:
    BATCH_SIZE: int = 1024
    EPOCHS: int = 200
    LEARNING_RATE: float = 0.0001
    WEIGHT_DECAY: float = 0.005
    DROPOUT_RATE: float = 0.4
    EARLY_STOPPING_PATIENCE: int = 20


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

    POSE_PAIRS_INDEXES: list[tuple[int, int]] = [
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

    NUM_ANGLES_PER_HAND: int = len(HAND_CONNECTIONS_INDEXES)
    # +1 para a distância do torso
    NUM_POSE_DISTANCES: int = len(POSE_PAIRS_INDEXES) + 1
    N_FEATURES: int = 2 * NUM_ANGLES_PER_HAND + NUM_POSE_DISTANCES
