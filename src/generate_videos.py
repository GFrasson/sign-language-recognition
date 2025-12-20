import os
import pickle
from glob import glob
from tqdm import tqdm
from visualization import create_skeleton_video
from entities.Settings import Settings


def main():
    features_path = Settings.FEATURES_PATH
    print(f"Scanning for keypoint files in {features_path}...")
    
    # We recursively find all pickle files that end with _keypoints.pkl
    pattern = os.path.join(features_path, "**", "*.pkl")
    keypoint_files = glob(pattern, recursive=True)
    
    print(f"Found {len(keypoint_files)} keypoint files.")
    
    for kp_file in tqdm(keypoint_files, desc="Generating videos"):
        try:
            with open(kp_file, 'rb') as f:
                data = pickle.load(f)
                
            landmarks = data.get('keypoints')
            if landmarks is None:
                print(f"No keypoints in {kp_file}")
                continue
                
            # output path: same name but .mp4 and _skeleton
            # e.g. video_keypoints.pkl -> video_skeleton.mp4
            output_path = kp_file.replace('.pkl', '.mp4')
            
            # Skip if already exists? Maybe not, user might want to regenerate.
            # if os.path.exists(output_path):
            #     continue

            create_skeleton_video(landmarks, output_path)
            
        except Exception as e:
            print(f"Error processing {kp_file}: {e}")

if __name__ == "__main__":
    main()
