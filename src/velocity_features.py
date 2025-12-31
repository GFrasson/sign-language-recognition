"""
Velocity and Acceleration Features for Sign Language Recognition.

This module extracts temporal dynamics features from landmark sequences:
- Velocity (rate of position change between frames)
- Acceleration (rate of velocity change)
- Movement direction and magnitude
- Kinematic signatures (peak velocity, movement phases)

These features help distinguish signs that have similar static positions
but different movement dynamics (e.g., "Perguntar" vs "Pesquisar").
"""
import numpy as np
from entities.Settings import LandmarkSettings


class VelocityFeaturesSettings:
    """Configuration for velocity feature extraction."""

    # Key landmarks to track for velocity (indices within each hand)
    # Wrist, Thumb tip, Index tip, Middle tip, Ring tip, Pinky tip
    HAND_KEY_POINTS: list[int] = [0, 4, 8, 12, 16, 20]

    # Pose key points for body movement tracking
    # Nose, Left wrist, Right wrist
    POSE_KEY_POINTS: list[int] = [0, 15, 16]

    # Number of features per hand: 6 points * 4 features (vel_mag, acc_mag, dir_x, dir_y)
    NUM_FEATURES_PER_HAND: int = len(HAND_KEY_POINTS) * 4

    # Number of features for pose: 3 points * 4 features
    NUM_POSE_FEATURES: int = len(POSE_KEY_POINTS) * 4

    # Global kinematic features: peak velocity frame, velocity variance, etc.
    NUM_KINEMATIC_FEATURES: int = 6

    # Total features per frame
    N_VELOCITY_FEATURES: int = (2 * NUM_FEATURES_PER_HAND) + NUM_POSE_FEATURES + NUM_KINEMATIC_FEATURES


def extract_velocity_features(landmarks_sequence: np.ndarray) -> np.ndarray:
    """
    Extract velocity and acceleration features from a landmark sequence.
    
    Args:
        landmarks_sequence: Array of shape (num_frames, total_landmarks * 3)
                           where total_landmarks = 543 (pose + face + 2 hands)
    
    Returns:
        Array of shape (num_frames, N_VELOCITY_FEATURES)
    """
    num_frames = landmarks_sequence.shape[0]
    
    # Reshape to (num_frames, num_landmarks, 3)
    all_landmarks = landmarks_sequence.reshape(num_frames, -1, 3)
    
    # Extract relevant landmark groups
    pose_landmarks = all_landmarks[:, LandmarkSettings.POSE_START:LandmarkSettings.POSE_END, :]
    left_hand_landmarks = all_landmarks[:, LandmarkSettings.LEFT_HAND_START:LandmarkSettings.LEFT_HAND_END, :]
    right_hand_landmarks = all_landmarks[:, LandmarkSettings.RIGHT_HAND_START:LandmarkSettings.RIGHT_HAND_END, :]
    
    # Extract key points for each group
    left_hand_key = left_hand_landmarks[:, VelocityFeaturesSettings.HAND_KEY_POINTS, :]  # (T, 6, 3)
    right_hand_key = right_hand_landmarks[:, VelocityFeaturesSettings.HAND_KEY_POINTS, :]
    pose_key = pose_landmarks[:, VelocityFeaturesSettings.POSE_KEY_POINTS, :]  # (T, 3, 3)
    
    # Calculate features for each group
    left_hand_vel_features = _compute_velocity_for_points(left_hand_key)
    right_hand_vel_features = _compute_velocity_for_points(right_hand_key)
    pose_vel_features = _compute_velocity_for_points(pose_key)
    
    # Compute global kinematic features
    kinematic_features = _compute_kinematic_signature(left_hand_key, right_hand_key)
    
    # Combine all features
    # Each velocity feature array has shape (T, num_points * 4)
    all_features = np.concatenate([
        left_hand_vel_features,
        right_hand_vel_features,
        pose_vel_features,
        kinematic_features
    ], axis=1)
    
    return all_features


def _compute_velocity_for_points(points: np.ndarray) -> np.ndarray:
    """
    Compute velocity and acceleration features for a set of tracked points.
    
    Args:
        points: Array of shape (num_frames, num_points, 3)
    
    Returns:
        Array of shape (num_frames, num_points * 4)
        Features per point: [velocity_magnitude, acceleration_magnitude, direction_x, direction_y]
    """
    num_frames, num_points, _ = points.shape
    
    # Handle missing hand (all zeros)
    if np.abs(points).sum() < 1e-6:
        return np.zeros((num_frames, num_points * 4))
    
    # Calculate velocity: difference between consecutive frames
    # Shape: (num_frames-1, num_points, 3)
    velocity = np.diff(points, axis=0)
    
    # Pad first frame with zeros (no velocity at t=0)
    velocity = np.concatenate([np.zeros((1, num_points, 3)), velocity], axis=0)
    
    # Calculate acceleration: difference of velocity
    acceleration = np.diff(velocity, axis=0)
    acceleration = np.concatenate([np.zeros((1, num_points, 3)), acceleration], axis=0)
    
    # Compute magnitudes
    velocity_mag = np.linalg.norm(velocity, axis=2)  # (T, num_points)
    acceleration_mag = np.linalg.norm(acceleration, axis=2)
    
    # Compute direction (normalized x, y components - ignoring z for 2D direction)
    velocity_norm = np.linalg.norm(velocity[:, :, :2], axis=2, keepdims=True)
    velocity_norm = np.where(velocity_norm < 1e-6, 1.0, velocity_norm)  # Avoid div by zero
    direction_xy = velocity[:, :, :2] / velocity_norm  # (T, num_points, 2)
    
    direction_x = direction_xy[:, :, 0]  # (T, num_points)
    direction_y = direction_xy[:, :, 1]
    
    # Normalize magnitudes by max to make them scale-invariant
    max_vel = np.max(velocity_mag) if np.max(velocity_mag) > 1e-6 else 1.0
    max_acc = np.max(acceleration_mag) if np.max(acceleration_mag) > 1e-6 else 1.0
    
    velocity_mag_norm = velocity_mag / max_vel
    acceleration_mag_norm = acceleration_mag / max_acc
    
    # Stack features: (T, num_points, 4) -> (T, num_points * 4)
    features = np.stack([velocity_mag_norm, acceleration_mag_norm, direction_x, direction_y], axis=2)
    features = features.reshape(num_frames, num_points * 4)
    
    return features


def _compute_kinematic_signature(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """
    Compute global kinematic features that characterize the movement pattern.
    
    Args:
        left_hand: Shape (num_frames, num_key_points, 3)
        right_hand: Shape (num_frames, num_key_points, 3)
    
    Returns:
        Array of shape (num_frames, NUM_KINEMATIC_FEATURES)
        Features: [peak_vel_frame, vel_variance, movement_symmetry, 
                   dominant_hand, acceleration_peaks, movement_duration_ratio]
    """
    num_frames = left_hand.shape[0]
    features = np.zeros((num_frames, VelocityFeaturesSettings.NUM_KINEMATIC_FEATURES))
    
    # Use index finger tip (point 2 in key points = landmark 8) as representative
    left_tip = left_hand[:, 2, :]  # (T, 3)
    right_tip = right_hand[:, 2, :]
    
    # Velocities
    left_vel = np.diff(left_tip, axis=0, prepend=left_tip[:1])
    right_vel = np.diff(right_tip, axis=0, prepend=right_tip[:1])
    
    left_vel_mag = np.linalg.norm(left_vel, axis=1)
    right_vel_mag = np.linalg.norm(right_vel, axis=1)
    
    # Combined velocity magnitude
    combined_vel = left_vel_mag + right_vel_mag
    
    # Feature 1: Normalized frame of peak velocity (0-1 scale)
    if np.max(combined_vel) > 1e-6:
        peak_frame = np.argmax(combined_vel) / max(num_frames - 1, 1)
    else:
        peak_frame = 0.5
    
    # Feature 2: Velocity variance (normalized)
    vel_variance = np.var(combined_vel) / (np.max(combined_vel) + 1e-6)
    
    # Feature 3: Movement symmetry (correlation between hands)
    if np.std(left_vel_mag) > 1e-6 and np.std(right_vel_mag) > 1e-6:
        symmetry = np.corrcoef(left_vel_mag, right_vel_mag)[0, 1]
        symmetry = 0.0 if np.isnan(symmetry) else symmetry
    else:
        symmetry = 0.0
    
    # Feature 4: Dominant hand indicator (-1 = left, +1 = right, 0 = balanced)
    total_left = np.sum(left_vel_mag)
    total_right = np.sum(right_vel_mag)
    if total_left + total_right > 1e-6:
        dominance = (total_right - total_left) / (total_right + total_left)
    else:
        dominance = 0.0
    
    # Feature 5: Number of acceleration peaks (normalized)
    acc = np.diff(combined_vel, prepend=combined_vel[:1])
    sign_changes = np.sum(np.diff(np.sign(acc)) != 0)
    acc_peaks = sign_changes / max(num_frames - 2, 1)
    
    # Feature 6: Movement duration ratio (frames with significant movement / total)
    threshold = 0.1 * np.max(combined_vel) if np.max(combined_vel) > 1e-6 else 0
    moving_frames = np.sum(combined_vel > threshold) / num_frames
    
    # Broadcast features to all frames
    features[:, 0] = peak_frame
    features[:, 1] = vel_variance
    features[:, 2] = symmetry
    features[:, 3] = dominance
    features[:, 4] = acc_peaks
    features[:, 5] = moving_frames
    
    return features


# Convenience function for getting feature count
def get_velocity_feature_count() -> int:
    """Returns the number of velocity features per frame."""
    return VelocityFeaturesSettings.N_VELOCITY_FEATURES
