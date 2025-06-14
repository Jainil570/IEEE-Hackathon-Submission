import cv2
import os
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

dataset_path = "signaura_dataset_dynamic"
os.makedirs(dataset_path, exist_ok=True)
gestures = ['Hello', 'goodbye', 'fu*k']
capture_interval = 0.1   
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

gesture_index = 0

while gesture_index < len(gestures):
    gesture_name = gestures[gesture_index]
    print(f"\nCollecting data for gesture: {gesture_name}")
    gesture_folder = os.path.join(dataset_path, gesture_name)
    os.makedirs(gesture_folder, exist_ok=True)
    
    capturing_sequence = False
    sequence_data = [] 
    last_capture_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Flip the frame for a mirrored view and convert to RGB
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Draw hand landmarks and capture data if sequence capture is enabled
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if capturing_sequence and time.time() - last_capture_time >= capture_interval:
                    # Extract landmarks: list of [x, y, z] for each of the 21 landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                    sequence_data.append(landmarks)
                    last_capture_time = time.time()
                    print(f"Captured frame {len(sequence_data)} for gesture '{gesture_name}'")
        else:
            # Optional: If no hand detected, you can choose to append a zero array or skip capture.
            if capturing_sequence and time.time() - last_capture_time >= capture_interval:
                sequence_data.append(np.zeros((21, 3), dtype=np.float32))
                last_capture_time = time.time()
                print(f"Captured empty frame {len(sequence_data)} for gesture '{gesture_name}'")
        
        # Display instructions on the frame
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to start, 'o' to stop, 'n' for next, 'q' to quit", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow("Dynamic Gesture Data Collection", frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if not capturing_sequence:
                capturing_sequence = True
                sequence_data = []  # Reset sequence data
                last_capture_time = time.time()
                print("Started capturing dynamic gesture sequence...")
        elif key == ord('a'):
            if capturing_sequence:
                capturing_sequence = False
                if sequence_data:
                    # Save the sequence as a numpy array file
                    timestamp = int(time.time())
                    filename = os.path.join(gesture_folder, f"{timestamp}.npy")
                    np.save(filename, np.array(sequence_data))
                    print(f"Saved dynamic gesture sequence: {filename}")
                    sequence_data = []  # Reset sequence data after saving
                else:
                    print("No data captured for this sequence.")
        elif key == ord('n'):
            print(f"Finished collecting data for gesture: {gesture_name}")
            gesture_index += 1  # Move to the next gesture
            break
        elif key == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
