import cv2
import numpy as np


def create_skeleton_video(landmarks: np.ndarray, output_path: str, width: int = 640, height: int = 480, fps: int = 5):
    """
    Generates a video with a white background and keypoints drawn as circles.
    
    Args:
        landmarks: Numpy array of shape (n_frames, 225) containing flattened landmarks (Pose+LH+RH).
        output_path: Path to save the video.
        width: Video width.
        height: Video height.
        fps: Frames per second.
    """
    if landmarks is None or len(landmarks) == 0:
        print(f"Skipping video generation for {output_path}: No landmarks found.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define ranges
    # Pose: 0-33 (33 points) -> 0 to 99 flat indices
    # LH: 33-54 (21 points) -> 99 to 162 flat indices
    # RH: 54-75 (21 points) -> 162 to 225 flat indices
    
    POSE_COUNT = 33
    HAND_COUNT = 21
    
    pose_end = POSE_COUNT * 3
    lh_end = pose_end + (HAND_COUNT * 3)
    
    for flat_frame in landmarks:
        # Create white image
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Helper to draw points
        def draw_points(points_flat, color):
            points = points_flat.reshape(-1, 3)
            for point in points:
                x, y, z = point
                # Check if point is not empty/zero (z can be 0 but x,y usually valid if detected)
                # MediaPipe returns normalized coordinates [0, 1]
                # If all zero, it wasn't detected (implemented as zeros in landmarks.py)
                if x == 0 and y == 0 and z == 0:
                    continue
                    
                px = int(x * width)
                py = int(y * height)
                
                cv2.circle(img, (px, py), 3, color, -1)

        # Draw Pose (Red)
        draw_points(flat_frame[:pose_end], (0, 0, 255))
        
        # Draw Left Hand (Green)
        draw_points(flat_frame[pose_end:lh_end], (0, 255, 0))
        
        # Draw Right Hand (Blue)
        draw_points(flat_frame[lh_end:], (255, 0, 0))

        out.write(img)

    out.release()
