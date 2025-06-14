import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.model_selection import train_test_split

# Define the dataset path and list of gesture folders
dataset_path = "signaura_dataset_dynamic"
gestures = sorted(os.listdir(dataset_path))
print("Gestures found:", gestures)

# Initialize lists to hold sequences and corresponding labels
X = []
y = []

# Load each sequence (.npy file) and assign a label based on its gesture folder
for label, gesture in enumerate(gestures):
    gesture_folder = os.path.join(dataset_path, gesture)
    for file in os.listdir(gesture_folder):
        if file.endswith('.npy'):
            sequence = np.load(os.path.join(gesture_folder, file))
            # Convert sequence to 2D if it's not already
            if sequence.ndim != 2:
                # Reshape: keep the first dimension (time) and flatten the rest
                sequence = sequence.reshape(sequence.shape[0], -1)
            if sequence.size > 0:
                X.append(sequence)
                y.append(label)

print(f"Loaded {len(X)} sequences.")

# Define a fixed sequence length (number of frames per sequence)
desired_seq_length = 50

# Determine the number of features from the first sequence (e.g., 21 landmarks * 3 = 63)
num_features = X[0].shape[1]

# Pad (or truncate) sequences to have the same length
X_fixed = []
for seq in X:
    if len(seq) < desired_seq_length:
        # Pad with zeros if sequence is too short
        pad_length = desired_seq_length - len(seq)
        padding = np.zeros((pad_length, num_features), dtype=np.float32)
        new_seq = np.concatenate([seq, padding], axis=0)
    else:
        # Truncate if sequence is too long
        new_seq = seq[:desired_seq_length]
    X_fixed.append(new_seq)

X = np.array(X_fixed)  # Now shape is (num_sequences, desired_seq_length, num_features)
y = np.array(y)

print("Final data shape:", X.shape)

# One-hot encode labels
num_classes = len(gestures)
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Build the GRU model
model = Sequential()
model.add(GRU(64, return_sequences=True, activation='relu', input_shape=X_train.shape[1:]))
model.add(GRU(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
epochs = 50
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

# Save the trained model
model.save("trainmodel_gru.h5")
print("Model saved as trainmodel_gru.h5")
