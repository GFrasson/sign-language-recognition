import os
import numpy as np
from tqdm import tqdm

from file_utils import get_folder_name, get_filename
from features import get_features
from dataset import Dataset


def process_videos(video_files: list[str], num_frames: int, save_dir: str, augment_factor: int = 20) -> Dataset:
    """Processa uma lista de vídeos, com aumento de dados."""
    X, y, signalers, is_augmented = [], [], [], []

    for video_file in tqdm(video_files, desc="Extraindo Features"):
        folder_name = get_folder_name(video_file)
        label, signaler, class_name = get_info_from_video_file(video_file)

        print(f"Processando: {video_file} -> Classe: {class_name} (ID: {label}), Sinalizador: {signaler}")

        features_save_dir_path = os.path.join(save_dir, folder_name)

        for i in range(augment_factor + 1):
            augment_index = i - 1 if i > 0 else None
            features = get_features(video_file, num_frames, label, signaler, features_save_dir_path, augment_index)

            if features is None:
                continue

            X.append(features)
            y.append(label)
            signalers.append(signaler)
            is_augmented.append(augment_index is not None)

    return Dataset(np.array(X), np.array(y), np.array(signalers), np.array(is_augmented))


def get_info_from_video_file(video_file: str) -> tuple[int, int, str]:
    folder_name = get_folder_name(video_file)
    filename = get_filename(video_file)

    label, class_name = folder_name.split('-')
    signaler = filename.split('-')[0]

    return int(label), int(signaler), class_name
