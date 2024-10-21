# Sign Language Gesture Recognition

This project is aimed at developing a system to help convert sign language gestures into text and voice, as well as converting text/voice back into sign language. It includes tools for collecting gesture data, processing it, and using machine learning for recognition and translation.

## Features

**Data Collection**: Capture images of hand signs using a webcam. The images are stored in a structured format for training machine learning models. Each image is                           labeled and stored with its corresponding person ID and image count for easy tracking.
**Sign to Text/Voice**: Recognize sign language gestures from images and convert them into text and speech using trained machine learning models.
**Text/Voice to Sign**: Convert text or spoken language into sign language gestures by displaying the corresponding sign images.

## Project Structure

```
/dataset
   /images              # Directory to store captured images
   /csv.csv             # CSV file to store image metadata (file name, label, and person ID)
README.md               # Project overview and setup instructions
image_capture.py        # Script to collect gesture data using a webcam
```

## How It Works

1. **Data Collection**: Use the `image_capture.py` script to collect images of hand gestures for various labels (e.g., numbers, alphabets). Each image is automatically named and stored along with its label and person ID in the `csv.csv` file.
2. **Model Training**: Use the captured images to train a machine learning model for recognizing sign language gestures.
3. **Recognition**: The system can convert recognized gestures into text and voice.
4. **Translation**: The system also supports translating text or voice into sign language gestures by displaying corresponding images.
