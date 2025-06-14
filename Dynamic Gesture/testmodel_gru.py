import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained GRU model
model = tf.keras.models.load_model(r"C:\Users\hp\Desktop\Jainil\dynamic gestures\zaidcode\trainmodel_gru.h5")

# Updated list of gestures (order should match training)
gestures = ['bzk', 'close', 'drink', 'goodbye', 'hello', 'rotate', 'walk']

# Set desired sequence length (same as used during training)
desired_seq_length = 50

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# List to store the captured sequence of hand landmarks
sequence = []

# Variable to hold the last high-confidence prediction
last_gesture = None
last_confidence = 0

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Flip the frame for a mirrored view and convert color space
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image to detect hand landmarks
    results = hands.process(image_rgb)

    # Initialize landmarks as an empty list
    landmarks = []
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Extract landmarks as a list of [x, y, z]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

    # If landmarks were detected, add to the sequence
    if landmarks:
        # Convert list to a NumPy array with shape (21, 3)
        landmarks = np.array(landmarks, dtype=np.float32)
        sequence.append(landmarks)

    # Once the sequence reaches the desired length, run prediction
    if len(sequence) == desired_seq_length:
        # Convert sequence to array and reshape:
        # From shape: (50, 21, 3) -> To shape: (1, 50, 63)
        input_sequence = np.array(sequence)
        input_sequence = input_sequence.reshape(1, desired_seq_length, -1)

        # Predict probabilities for each gesture
        prediction = model.predict(input_sequence)
        predicted_index = np.argmax(prediction)
        predicted_gesture = gestures[predicted_index]
        confidence = np.max(prediction)

        # If the confidence is above 60%, update the last recognized gesture.
        if confidence >= 0.7:
            last_gesture = predicted_gesture
            last_confidence = confidence
            print(f"New Prediction: {predicted_gesture} with confidence {confidence*100:.1f}%")
        else:
            print(f"Low confidence ({confidence*100:.1f}%), keeping previous gesture.")

        # Clear the sequence for the next prediction
        sequence = []

    # If a gesture has been recognized with high confidence, display it on the frame.
    if last_gesture is not None:
        cv2.putText(frame, f"Gesture: {last_gesture} ({last_confidence*100:.1f}%)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the webcam feed with the overlay
    cv2.imshow("Dynamic Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
