import os
import pickle

from geometric_features import extract_custom_geometric_features
from landmarks import process_frames
from file_utils import make_directories, get_filename
from entities.Settings import GeometricFeaturesSettings


def get_features_from_frames(video_frames, video_file, label, signaler, features_save_dir_path, augment_index=None):
    """
    Versão otimizada que recebe frames já carregados para evitar re-leitura de disco.
    """
    features = load_features(video_file, features_save_dir_path, augment_index)
    if features is not None:
        return features

    return extract_features_from_frames(video_frames, video_file, label, signaler, features_save_dir_path, augment_index)


def extract_features_from_frames(video_frames, video_file, label, signaler, features_save_dir_path: str, augment_index: int = None):
    """
    Extrai features a partir de frames em memória.
    """
    landmarks = process_frames(video_frames, augment=augment_index is not None)

    if landmarks is None or landmarks.shape[0] != len(video_frames):
        # Note: if process_frames fails somehow
        print(f"Erro ao processar frames de: {video_file}")
        return None

    geometric_features = extract_custom_geometric_features(landmarks)

    save_features(geometric_features, landmarks, label, signaler, video_file, features_save_dir_path, augment_index)

    return geometric_features


def save_features(features, landmarks, label, signaler, video_file: str, save_dir: str, augment_index: int = None) -> str:
    """Salva features, label e sinalizador em um arquivo .pkl."""
    make_directories(save_dir)

    feature_filename = build_features_filename(video_file, augment_index)
    save_path = os.path.join(save_dir, feature_filename)

    with open(save_path, 'wb') as file:
        pickle.dump({
            'keypoints': landmarks,
            'features': features,
            'label': label,
            'signaler': signaler
        }, file)

    return save_path


def load_features(video_file: str, save_dir: str, augment_index: int = None):
    """Carrega features salvas, se existirem."""
    features_filename = build_features_filename(video_file, augment_index)
    save_path = os.path.join(save_dir, features_filename)

    if os.path.exists(save_path):
        data = {}
        with open(save_path, 'rb') as file:
            data = pickle.load(file)

        features = data['features']

        # Check for dimension mismatch (Legacy vs New features)
        if features is not None and features.shape[1] != GeometricFeaturesSettings.N_FEATURES:
            # print(f"Dimension mismatch for {video_file}: Loaded {features.shape[1]}, Expected {GeometricFeaturesSettings.N_FEATURES}. Attempting to re-calculate from keypoints.")
            
            if 'keypoints' in data and data['keypoints'] is not None:
                # Re-calculate features using current settings (e.g. Legacy)
                features = extract_custom_geometric_features(data['keypoints'])
            
        return features

    return None


def build_features_filename(video_file: str, augment_index: int = None) -> str:
    filename = get_filename(video_file)
    base_name = os.path.splitext(filename)[0]

    if augment_index is None:
        return f"{base_name}_features.pkl"

    return f"{base_name}_features_aug_{augment_index}.pkl"
