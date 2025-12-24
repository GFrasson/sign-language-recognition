import concurrent.futures
import os
import numpy as np
from tqdm import tqdm

from file_utils import get_folder_name, get_filename
from entities.Dataset import Dataset
from class_mapping import class_name_mapping
from features import load_features, get_features_from_frames
from landmarks import read_all_video_frames, sample_frames_from_list
from entities.Settings import GeometricFeaturesSettings


def process_videos(video_files: list[str], num_frames: int, save_dir: str, augment_factor: int = 20, use_legacy: bool = False) -> Dataset:
    """Processa uma lista de vídeos, com aumento de dados. (Paralelizado)"""
    X, y, signalers, is_augmented = [], [], [], []
    class_map = {}

    # Prepare arguments for parallel processing
    tasks = [(video_file, num_frames, save_dir, augment_factor, use_legacy) for video_file in video_files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        results = list(tqdm(executor.map(process_single_video_wrapper, tasks), total=len(video_files), desc="Extraindo Features"))

    for result in results:
        if result is None:
            continue
        
        video_features_list, label, signaler, class_name = result
        
        if label not in class_map:
            class_map[label] = class_name
            
        for features, is_aug in video_features_list:
            X.append(features)
            y.append(label)
            signalers.append(signaler)
            is_augmented.append(is_aug)

    return Dataset(np.array(X), np.array(y), np.array(signalers), class_map, np.array(is_augmented))


def process_single_video_wrapper(args):
    """Wrapper para desempacotar argumentos para o map do executor."""
    # Desempacotar incluindo use_legacy que é o último argumento
    video_file, num_frames, save_dir, augment_factor, use_legacy = args
    
    # Configurar Settings no processo filho (crucial para Windows/spawn)
    GeometricFeaturesSettings.configure(use_legacy)
    
    return process_single_video(video_file, num_frames, save_dir, augment_factor)


def process_single_video(video_file: str, num_frames: int, save_dir: str, augment_factor: int):
    """Processa um único vídeo e suas augumentações."""
    try:
        folder_name = get_folder_name(video_file)
        label, signaler, class_name = get_info_from_video_file(video_file)
        
        features_save_dir_path = os.path.join(save_dir, folder_name)
        
        video_features_list = []
        
        # 1. First, check which ones serve from cache
        missing_indices = []
        
        # Check original
        feat_orig = load_features(video_file, features_save_dir_path, None)
        if feat_orig is not None:
            video_features_list.append((feat_orig, False))
        else:
            missing_indices.append(None)
             
        # Check augments
        for i in range(1, augment_factor + 1):
            aug_idx = i - 1
            feat_aug = load_features(video_file, features_save_dir_path, aug_idx)
            if feat_aug is not None:
                video_features_list.append((feat_aug, True))
            else:
                missing_indices.append(aug_idx)
        
        # If no missing indices, we are done
        if not missing_indices:
            return video_features_list, label, signaler, class_name

        # 2. If we have missing features, READ ALL VIDEO FRAMES ONCE (High RAM usage)
        
        # Read ALL frames into memory
        all_video_frames = read_all_video_frames(video_file)
        
        if all_video_frames is None:
            # If video fails to read, reuse what we have in cache if any
            if video_features_list:
                return video_features_list, label, signaler, class_name
            return None

        # 3. Process missing indices
        for idx in missing_indices:
            # Sample specific frames for this iteration to ensure temporal diversity
            # Each iteration gets a different random subset of frames (normal distribution)
            sampled_frames = sample_frames_from_list(all_video_frames, num_frames)
            
            if sampled_frames is None:
                print(f"Error sampling frames for {video_file} (Total: {len(all_video_frames)})")
                continue
                
            feat = get_features_from_frames(sampled_frames, video_file, label, signaler, features_save_dir_path, idx)
            if feat is not None:
                video_features_list.append((feat, idx is not None))
        
        if not video_features_list:
            return None

        return video_features_list, label, signaler, class_name
    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        return None


def get_info_from_video_file(video_file: str) -> tuple[int, int, str]:
    folder_name = get_folder_name(video_file)
    filename = get_filename(video_file)

    label = folder_name.split('-')[0]
    signaler = filename.split('-')[0]

    class_name = format_class_name(int(label))

    return int(label), int(signaler), class_name


def format_class_name(label: int) -> str:
    """Formata o nome da classe para exibição (acentos, espaços, etc)."""
    return class_name_mapping.get(label) or ""
