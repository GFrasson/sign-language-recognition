import os
import numpy as np
from tqdm import tqdm

from landmarks import extract_landmarks
from geometric_features import extract_custom_geometric_features


def list_video_files(data_path):
    """Busca todos os vídeos MP4 nas subpastas do diretório."""
    video_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    return video_files


def extract_features_and_labels(video_files, class_mapping, num_frames):
    """Extrai features geométricas e rótulos de uma lista de vídeos."""
    X, y = [], []
    for video_file in tqdm(video_files, desc="Extraindo Features"):
        folder_name = os.path.basename(os.path.dirname(video_file))

        if folder_name not in class_mapping:
            print(f"Classe não encontrada no mapeamento: {folder_name}")
            continue

        label = class_mapping[folder_name]
        print(f"Processando: {video_file} -> Classe: {folder_name} (ID: {label})")

        raw_landmarks = extract_landmarks(video_file, num_frames)
        if raw_landmarks is None or raw_landmarks.shape[0] != num_frames:
            print(f"Erro ao processar vídeo: {video_file}")
            continue

        features = extract_custom_geometric_features(raw_landmarks)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)
