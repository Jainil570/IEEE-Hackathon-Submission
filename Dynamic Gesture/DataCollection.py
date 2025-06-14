import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Define output folder for collected data
DATA_DIR = "gesture_data"
GESTURE_LABELS = ["hello", "thanks", "yes", "no"]  # Define gesture labels
NUM_SAMPLES = 100  # Number of samples per gesture

# Create directories for each gesture
for label in GESTURE_LABELS:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

# Capture hand landmarks
cap = cv2.VideoCapture(0)
for label in GESTURE_LABELS:
    print(f"Collecting data for {label}. Press 's' to start and 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                # Save landmarks
                file_path = os.path.join(DATA_DIR, label, f"{len(os.listdir(os.path.join(DATA_DIR, label)))}.npy")
                np.save(file_path, landmarks)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
