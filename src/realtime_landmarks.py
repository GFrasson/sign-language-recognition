import cv2
import mediapipe as mp
from landmarks import create_holistic_model

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

        cv2.imshow('Webcam - Landmarks (Press Q to quit)', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_landmarks()
