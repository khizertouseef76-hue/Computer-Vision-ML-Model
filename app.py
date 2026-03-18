import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/emotion_model.h5")
    return model

model = load_model()

# Load face detector
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
emotion_labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]
IMG_SIZE = 96

# -------------------- UI Layout --------------------
st.set_page_config(page_title="Emotion Detector AI", layout="wide")
st.title("🎭 Real-Time Emotion Detector")
st.write("Detect emotions from uploaded images or live webcam.")

mode = st.radio("Select Mode:", ["Upload Image", "Live Webcam"])

# -------------------- IMAGE UPLOAD MODE --------------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload a selfie or photo", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img_array[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            predictions = model.predict(face, verbose=0)
            emotion_idx = np.argmax(predictions)
            emotion = emotion_labels[emotion_idx]
            confidence = predictions[0][emotion_idx]

            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img_array, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        st.image(img_array, channels="BGR", caption="Processed Image")

# -------------------- LIVE WEBCAM MODE --------------------
elif mode == "Live Webcam":
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    stop_button = st.button("Stop Camera")

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Could not access webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            predictions = model.predict(face, verbose=0)
            emotion_idx = np.argmax(predictions)
            emotion = emotion_labels[emotion_idx]
            confidence = predictions[0][emotion_idx]

            # Draw rectangle + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

        # Stop if button pressed
        if stop_button:
            break

    camera.release()
    cv2.destroyAllWindows()
