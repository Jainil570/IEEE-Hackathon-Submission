import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Dropout, TimeDistributed
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load collected data
DATA_DIR = "gesture_data"
gestures = []
labels = []

for label in os.listdir(DATA_DIR):
    for file in os.listdir(os.path.join(DATA_DIR, label)):
        file_path = os.path.join(DATA_DIR, label, file)
        landmarks = np.load(file_path)

        gestures.append(landmarks)
        labels.append(label)

# Convert to NumPy arrays
gestures = np.array(gestures)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(gestures, labels_encoded, test_size=0.2, random_state=42)

# Ensure input shape is 3D (batch_size, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)

# Define CNN + LSTM model
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),

    # TimeDistributed to preserve time steps before LSTM
    TimeDistributed(Dense(64, activation="relu")),

    LSTM(64, return_sequences=True),
    LSTM(32),

    Dense(32, activation="relu"),
    Dropout(0.2),
    
    Dense(len(label_encoder.classes_), activation="softmax")
])

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("signaura_gesture_model.h5")
