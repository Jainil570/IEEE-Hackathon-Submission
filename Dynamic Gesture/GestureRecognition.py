import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = load_model("signaura_gesture_model.h5")

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["hello", "thanks", "yes", "no"])  # Ensure these match training labels

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Reshape input for model
            landmarks = landmarks.reshape(1, landmarks.shape[0], 1)

            # Predict gesture
            prediction = model.predict(landmarks)
            class_index = np.argmax(prediction)
            gesture_label = label_encoder.inverse_transform([class_index])[0]

            # Display prediction
            cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
