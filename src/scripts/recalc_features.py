import os
import pickle
from tqdm import tqdm
from entities.Settings import Settings, GeometricFeaturesSettings
from geometric_features import extract_custom_geometric_features

def recalc_all():
    # Source directory (hardceded as per request)
    SOURCE_DIR = "data/features-hands-distances-normal"
    # Target directory (from Settings)
    TARGET_DIR = Settings.FEATURES_PATH 
    
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    
    # Configure for NEW features
    GeometricFeaturesSettings.configure(use_legacy=False)
    
    # Check if we are running the right logic
    print(f"N_FEATURES: {GeometricFeaturesSettings.N_FEATURES}")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    # Walk through source directory
    files_to_process = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith(".pkl"):
                files_to_process.append(os.path.join(root, file))
    
    print(f"Found {len(files_to_process)} files.")
    
    for file_path in tqdm(files_to_process, desc="Recalculating Features"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            keypoints = data['keypoints']
            label = data['label']
            signaler = data['signaler']
            
            # Recalculate
            new_features = extract_custom_geometric_features(keypoints)
            
            # Determine relative path to maintain structure
            rel_path = os.path.relpath(file_path, SOURCE_DIR)
            target_path = os.path.join(TARGET_DIR, rel_path)
            
            # Ensure dir exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Save
            with open(target_path, 'wb') as f:
                 pickle.dump({
                    'keypoints': keypoints,
                    'features': new_features,
                    'label': label,
                    'signaler': signaler
                }, f)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    recalc_all()
