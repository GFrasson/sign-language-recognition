import os
import pickle
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


def save_features(features, label, signaler, video_file, save_dir):
    """Salva features, label e sinalizador em um arquivo .pkl."""
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(video_file)
    base_name = os.path.splitext(filename)[0]
    save_path = os.path.join(save_dir, f"{base_name}_features.pkl")

    with open(save_path, 'wb') as file:
        pickle.dump({
            'features': features,
            'label': label,
            'signaler': signaler
        }, file)

    return save_path


def load_features(video_file, save_dir):
    """Carrega features salvas, se existirem."""
    filename = os.path.basename(video_file)
    base_name = os.path.splitext(filename)[0]
    save_path = os.path.join(save_dir, f"{base_name}_features.pkl")

    if os.path.exists(save_path):
        data = {}
        with open(save_path, 'rb') as file:
            data = pickle.load(file)

        return data['features'], data['label'], data['signaler']
    
    return None, None, None


def extract_features_and_labels(video_files, num_frames, save_dir):
    """Extrai features geométricas, rótulos e sinalizadores de uma lista de vídeos."""
    X, y, signalers = [], [], []
    for video_file in tqdm(video_files, desc="Extraindo Features"):
        folder_name = os.path.basename(os.path.dirname(video_file))

        features_save_dir_path = os.path.join(save_dir, folder_name)
        features, label, signaler = load_features(video_file, features_save_dir_path)

        if features is not None:
            X.append(features)
            y.append(label)
            signalers.append(signaler)
            continue

        filename = os.path.basename(video_file)
        label, class_name = folder_name.split('-')
        signaler = filename.split('-')[0]

        try:
            label = int(label)
            signaler = int(signaler)
        except ValueError:
            print(f"Erro ao converter o label da classe ou do sinalizador para inteiro: {label}")
            continue

        print(f"Processando: {video_file} -> Classe: {class_name} (ID: {label}), Sinalizador: {signaler}")

        raw_landmarks = extract_landmarks(video_file, num_frames)
        if raw_landmarks is None or raw_landmarks.shape[0] != num_frames:
            print(f"Erro ao processar vídeo: {video_file}")
            continue

        features = extract_custom_geometric_features(raw_landmarks)

        X.append(features)
        y.append(label)
        signalers.append(signaler)
        
        save_features(features, label, signaler, video_file, features_save_dir_path)

    return np.array(X), np.array(y), np.array(signalers)
