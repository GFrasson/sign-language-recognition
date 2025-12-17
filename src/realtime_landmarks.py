import cv2
import mediapipe as mp
import numpy as np
from landmarks import create_holistic_model
from geometric_features import extract_frame_features
from entities.Settings import GeometricFeaturesSettings 

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def run_realtime_landmarks():
    holistic_model = create_holistic_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(frame_rgb)

        annotated_frame = frame_resized.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # if results.face_landmarks:
        #     mp_drawing.draw_landmarks(
        #         annotated_frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract raw landmarks for feature calculation
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 3)
        face = np.zeros(468 * 3)  # Not used for geometric features
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)

        frame_landmarks = np.concatenate([pose, face, lh, rh])
        features = extract_frame_features(frame_landmarks)

        # Display first 5 hand angles and first 5 pose distances with connection names
        left_hand_angles = features[:26]
        right_hand_angles = features[26:52]
        pose_distances = features[52:]

        y0 = 20
        dy = 20
        for i in range(5):
            lh_conn = GeometricFeaturesSettings.HAND_CONNECTIONS_INDEXES[i]
            rh_conn = GeometricFeaturesSettings.HAND_CONNECTIONS_INDEXES[i]
            pose_conn = GeometricFeaturesSettings.POSE_PAIRS_INDEXES[i] if i < len(GeometricFeaturesSettings.POSE_PAIRS_INDEXES) else ("-", "-")
            lh_label = f"LH Angle {i+1} ({lh_conn[0]}-{lh_conn[1]}-{lh_conn[2]})"
            rh_label = f"RH Angle {i+1} ({rh_conn[0]}-{rh_conn[1]}-{rh_conn[2]})"
            pose_label = f"Pose Dist {i+1} ({pose_conn[0]}-{pose_conn[1]})"
            cv2.putText(annotated_frame, f"{lh_label}: {left_hand_angles[i]:.1f}", (10, y0 + i*dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(annotated_frame, f"{rh_label}: {right_hand_angles[i]:.1f}", (250, y0 + i*dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(annotated_frame, f"{pose_label}: {pose_distances[i]:.2f}", (500, y0 + i*dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        cv2.imshow('Webcam - Landmarks & Features (Press Q to quit)', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_landmarks()