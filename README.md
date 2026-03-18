🎭 Emotion Detection AI

A deep learning–based facial emotion recognition system built using TensorFlow, OpenCV, and Streamlit.
This application can detect human emotions from:

📁 Uploaded images

🎥 Live webcam feed

🚀 Features

Real-time emotion detection

Face detection using Haar Cascade

CNN model trained on facial expression dataset

Clean browser-based UI using Streamlit

Confidence score display for predictions

🧠 Supported Emotions

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

📁 Project Structure
CV ML Model/
│
├── data/
│    ├── fer2013.csv
│    ├── Training/
│    ├── PublicTest/
│    └── PrivateTest/
│
├── models/
│    ├── emotion_model.h5
│    └── haarcascade_frontalface_default.xml
│
├── src/
│    ├── prepare_data.py
│    ├── train_model.py
│    └── streamlit_app.py
│
├── venv/
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Create Virtual Environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

2️⃣ Install Dependencies
pip install tensorflow opencv-python streamlit pillow numpy


Or if you have a requirements.txt:

pip install -r requirements.txt

  Train the Model

Run:

python src/train_model.py


This will generate:

models/emotion_model.h5

🌐 Run the App

From the project root:

streamlit run src/streamlit_app.py


The app will open automatically in your browser.

📌 Requirements

Python 3.9+

Webcam (for live detection)

Trained model file inside /models


👨‍💻 Author
Khizer Ahmad
Built as a computer vision deep learning project.