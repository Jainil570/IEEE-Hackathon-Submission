import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load landmark data
data = []
labels = []
classes = os.listdir("Data")

for idx, class_name in enumerate(classes):
    path = f"Data/{class_name}"
    for landmark_file in os.listdir(path):
        landmark = np.load(os.path.join(path, landmark_file))
        data.append(landmark.flatten())
        labels.append(idx)

data = np.array(data)
labels = np.array(labels)

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='linear',probability=True)
model.fit(x_train, y_train)

# Evaluate model
accuracy = model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open("New_gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)