import pandas as pd
import numpy as np
import cv2
import os

# Emotion labels
emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

DATA_PATH = "data/fer2013.csv"
BASE_OUTPUT_DIR = "data"
IMG_SIZE = 96

print("Loading CSV...")
df = pd.read_csv(DATA_PATH)

print("Processing images...")

for index, row in df.iterrows():
    emotion = emotion_map[row["emotion"]]
    usage = row["Usage"]  # Training / PublicTest / PrivateTest

    pixels = np.array(row["pixels"].split(), dtype="uint8")
    image = pixels.reshape(48, 48)

    # Convert grayscale → RGB
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize to 96x96
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Create folder structure
    save_dir = os.path.join(BASE_OUTPUT_DIR, usage, emotion)
    os.makedirs(save_dir, exist_ok=True)

    # Save image
    cv2.imwrite(os.path.join(save_dir, f"{index}.jpg"), image)

print("Dataset prepared successfully.")
