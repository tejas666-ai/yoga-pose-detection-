import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from math import degrees, acos

# Load the trained model
model = joblib.load("yoga_pose_classifier.joblib")

# Define required angle features
feature_names = [
    'left_elbow_angle', 'right_elbow_angle',
    'left_shoulder_angle', 'right_shoulder_angle',
    'left_knee_angle', 'right_knee_angle',
    'angle_for_ardhaChandrasana1', 'angle_for_ardhaChandrasana2',
    'hand_angle', 'left_hip_angle', 'right_hip_angle',
    'neck_angle_uk', 'left_wrist_angle_bk', 'right_wrist_angle_bk'
]

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Utility function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Extract required angles from pose landmarks
def extract_angles(landmarks):
    try:
        angles = {
            'left_elbow_angle': calculate_angle(landmarks[11], landmarks[13], landmarks[15]),
            'right_elbow_angle': calculate_angle(landmarks[12], landmarks[14], landmarks[16]),
            'left_shoulder_angle': calculate_angle(landmarks[13], landmarks[11], landmarks[23]),
            'right_shoulder_angle': calculate_angle(landmarks[14], landmarks[12], landmarks[24]),
            'left_knee_angle': calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
            'right_knee_angle': calculate_angle(landmarks[24], landmarks[26], landmarks[28]),
            'angle_for_ardhaChandrasana1': calculate_angle(landmarks[11], landmarks[23], landmarks[25]),
            'angle_for_ardhaChandrasana2': calculate_angle(landmarks[12], landmarks[24], landmarks[26]),
            'hand_angle': calculate_angle(landmarks[15], landmarks[13], landmarks[11]),
            'left_hip_angle': calculate_angle(landmarks[11], landmarks[23], landmarks[25]),
            'right_hip_angle': calculate_angle(landmarks[12], landmarks[24], landmarks[26]),
            'neck_angle_uk': calculate_angle(landmarks[11], landmarks[0], landmarks[12]),
            'left_wrist_angle_bk': calculate_angle(landmarks[13], landmarks[15], landmarks[17]),
            'right_wrist_angle_bk': calculate_angle(landmarks[14], landmarks[16], landmarks[18])
        }
        return angles
    except:
        return None

# Streamlit App
st.title("🧘 Real-Time Yoga Pose Classifier")
run = st.toggle("Start Camera")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark
                landmark_points = [(lm.x, lm.y, lm.z) for lm in landmarks]
                angles = extract_angles(landmark_points)

                if angles:
                    input_df = pd.DataFrame([angles])[feature_names]
                    prediction = model.predict(input_df)[0]
                    cv2.putText(frame, f'Pose: {prediction}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
else:
    st.warning("Camera stopped.")
