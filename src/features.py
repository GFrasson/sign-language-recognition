import os
import pickle

from geometric_features import extract_custom_geometric_features
from landmarks import extract_landmarks
from file_utils import make_directories, get_filename


def get_features(video_file: str, num_frames: int, label: int, signaler: int, features_save_dir_path: str, augment_index: int = None):
    features = load_features(video_file, features_save_dir_path, augment_index)

    if features is not None:
        return features

    return extract_features(video_file, num_frames, label, signaler, features_save_dir_path, augment_index)    


def extract_features(video_file: str, num_frames: int, label: int, signaler: int, features_save_dir_path: str, augment_index: int = None):
    landmarks = extract_landmarks(video_file, num_frames, augment=augment_index is not None)

    if landmarks is None or landmarks.shape[0] != num_frames:
        print(f"Erro ao processar vídeo: {video_file}")
        return None

    geometric_features = extract_custom_geometric_features(landmarks)

    save_features(geometric_features, label, signaler, video_file, features_save_dir_path, augment_index)

    return geometric_features


def save_features(features, label, signaler, video_file: str, save_dir: str, augment_index: int = None) -> str:
    """Salva features, label e sinalizador em um arquivo .pkl."""
    make_directories(save_dir)

    feature_filename = build_features_filename(video_file, augment_index)
    save_path = os.path.join(save_dir, feature_filename)

    with open(save_path, 'wb') as file:
        pickle.dump({
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

        # return data['features'], data['label'], data['signaler']
        return data['features']

    return None


def build_features_filename(video_file: str, augment_index: int = None) -> str:
    filename = get_filename(video_file)
    base_name = os.path.splitext(filename)[0]

    if augment_index is None:
        return f"{base_name}_features.pkl"

    return f"{base_name}_features_aug_{augment_index}.pkl"
