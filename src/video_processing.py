import os
import pickle
import numpy as np
from tqdm import tqdm

from landmarks import LandmarkExtractor
from geometric_features import extract_custom_geometric_features


def list_video_files(data_path):
    """Busca todos os vídeos MP4 nas subpastas do diretório."""
    video_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    return video_files


def save_features(features, label, signaler, video_file, save_dir, augment_index=None):
    """Salva features, label e sinalizador em um arquivo .pkl."""
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(video_file)
    base_name = os.path.splitext(filename)[0]
    feature_filename = f"{base_name}_features.pkl" if augment_index is None else f"{base_name}_features_aug_{augment_index}.pkl"
    save_path = os.path.join(save_dir, feature_filename)

    with open(save_path, 'wb') as file:
        pickle.dump({
            'features': features,
            'label': label,
            'signaler': signaler
        }, file)

    return save_path


def load_features(video_file, save_dir, augment_index=None):
    """Carrega features salvas, se existirem."""
    filename = os.path.basename(video_file)
    base_name = os.path.splitext(filename)[0]
    feature_filename = f"{base_name}_features.pkl" if augment_index is None else f"{base_name}_features_aug_{augment_index}.pkl"
    save_path = os.path.join(save_dir, feature_filename)

    if os.path.exists(save_path):
        data = {}
        with open(save_path, 'rb') as file:
            data = pickle.load(file)

        # return data['features'], data['label'], data['signaler']
        return data['features']
    
    return None


def extract_features_and_labels(video_files, num_frames, save_dir, augment_factor=20):
    """Extrai features geométricas, rótulos e sinalizadores de uma lista de vídeos, com aumento de dados."""
    X, y, signalers = [], [], []
    landmarks_extractor = LandmarkExtractor()

    for video_file in tqdm(video_files, desc="Extraindo Features"):
        folder_name = os.path.basename(os.path.dirname(video_file))

        filename = os.path.basename(video_file)
        label, class_name = folder_name.split('-')
        signaler = filename.split('-')[0]

        try:
            label = int(label)
            signaler = int(signaler)
        except ValueError:
            print(f"Erro ao converter o label da classe ou do sinalizador para inteiro: {label}")
            continue
        
        features_save_dir_path = os.path.join(save_dir, folder_name)
        features = load_features(video_file, features_save_dir_path)

        if features is None:
            print(f"Processando: {video_file} -> Classe: {class_name} (ID: {label}), Sinalizador: {signaler}")

            # Extração original (sem aumento)
            raw_landmarks = landmarks_extractor.extract_landmarks(video_file, num_frames)
            if raw_landmarks is None or raw_landmarks.shape[0] != num_frames:
                print(f"Erro ao processar vídeo: {video_file}")
                continue

            features = extract_custom_geometric_features(raw_landmarks)
            save_features(features, label, signaler, video_file, features_save_dir_path)
        
        X.append(features)
        y.append(label)
        signalers.append(signaler)
        
        # Aumento de dados (até augment_factor vezes)
        for i in range(augment_factor):
            features_save_dir_path = os.path.join(save_dir, folder_name)
            features = load_features(video_file, features_save_dir_path, augment_index=i)

            if features is not None:
                X.append(features)
                y.append(label)
                signalers.append(signaler)
                continue

            raw_landmarks_aug = landmarks_extractor.extract_landmarks(video_file, num_frames, augment=True)
            if raw_landmarks_aug is None or raw_landmarks_aug.shape[0] != num_frames:
                print(f"Erro ao processar vídeo aumentado: {video_file} (iter {i})")
                continue

            features_aug = extract_custom_geometric_features(raw_landmarks_aug)
            
            X.append(features_aug)
            y.append(label)
            signalers.append(signaler)

            save_features(features_aug, label, signaler, video_file, features_save_dir_path, augment_index=i)

    return np.array(X), np.array(y), np.array(signalers)
