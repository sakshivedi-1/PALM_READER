# PALM_READER
AI-powered palmistry app that reads palm lines from uploaded images and predicts personality traits and future insights using computer vision and machine learning.

 ## Project Overview
This project analyzes palm images using OpenCV and a pre-trained ML model to predict palmistry features such as heart line, head line, and life line. The application is deployed using Streamlit, offering an interactive front-end experience.

## Tech Stack
Frontend: Streamlit

Backend: Python, OpenCV, dlib, NumPy

ML Model: Custom palm-line classifier (pre-trained)

Deployment: Localhost / Streamlit Cloud (optionally)

## How to Run Locally
 Clone the Repository
bash
```
git clone https://github.com/sakshivedi-1/PALM_READER.git
cd PALM_READER
```
 Set Up Virtual Environment (Optional but Recommended)
bash
```
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```
 Install Required Packages
bash
```
pip install -r requirements.txt
```
If requirements.txt is not available, manually install:

bash
```
pip install streamlit opencv-python dlib numpy Pillow
```
Run the Application
bash
```
streamlit run app.py
```
Then open the provided localhost URL in your browser (e.g. http://localhost:8501).

## Project Structure
bash
```
PALM_READER/

│
├── app.py                 # Streamlit frontend

├── palm_reader.py         # Core ML + image processing logic

├── utils.py               # Helper functions

├── sample_images/         # Demo palm images

├── models/                # Trained ML models 

├── requirements.txt       # Dependencies

└── README.md              # This file
```
## How It Works
Put your hand in front of camera.

Make sure background should be solid colour.

The app detects key lines (heart, head, life).

Displays prediction or insights based on palmistry logic.

 


