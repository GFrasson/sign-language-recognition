import os
import numpy as np
from tqdm import tqdm

from file_utils import get_folder_name, get_filename
from features import get_features


def process_videos(video_files: list[str], num_frames: int, save_dir: str, augment_factor: int = 20):
    """Processa uma lista de vídeos, com aumento de dados."""
    X, y, signalers = [], [], []

    for video_file in tqdm(video_files, desc="Extraindo Features"):
        print(f"Processando: {video_file} -> Classe: {class_name} (ID: {label}), Sinalizador: {signaler}")

        folder_name = get_folder_name(video_file)
        label, signaler, class_name = get_info_from_video_file(video_file)

        features_save_dir_path = os.path.join(save_dir, folder_name)

        for i in range(augment_factor + 1):
            augment_index = i - 1 if i > 0 else None
            features = get_features(video_file, num_frames, label, signaler, features_save_dir_path, augment_index)

            if features is None:
                continue

            X.append(features)
            y.append(label)
            signalers.append(signaler)

    return np.array(X), np.array(y), np.array(signalers)


def get_info_from_video_file(video_file: str) -> tuple[int, int, str]:
    folder_name = get_folder_name(video_file)
    filename = get_filename(video_file)

    label, class_name = folder_name.split('-')
    signaler = filename.split('-')[0]

    return int(label), int(signaler), class_name
