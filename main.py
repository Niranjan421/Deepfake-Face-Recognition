# Import necessary libraries
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image



# Function to extract frames from video
def extract_frames(video_path, frame_rate=1):
    """
    Extract frames from a video file.

    Args:
    - video_path (str): Path to the video file.
    - frame_rate (int): Frame rate to extract frames at. Defaults to 1.

    Returns:
    - frames (numpy array): Array of extracted frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)  # Extract one frame per second by default
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval == 0:
            frame = cv2.resize(frame, (224, 224))  # Resizing frames to 224x224 for ResNet50 input
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

# Function to preprocess frames for ResNet50
def preprocess_frames(frames):
    """
    Preprocess frames for ResNet50 input.

    Args:
    - frames (numpy array): Array of frames to preprocess.

    Returns:
    - preprocessed_frames (numpy array): Preprocessed frames.
    """
    frames = frames.astype('float32') / 255.0  # Normalize to [0, 1]
    return frames

# Function to predict whether the video is real or fake
def predict_deepfake(frames, model):
    """
    Predict whether a video is real or fake using a trained ResNet50 model.

    Args:
    - frames (numpy array): Array of preprocessed frames.
    - model (keras model): Trained ResNet50 model.

    Returns:
    - prediction (float): Prediction score (0.0 - 1.0).
    """
    predictions = model.predict(frames)
    average_prediction = np.mean(predictions)  # Averaging predictions across frames
    return average_prediction



#SIDEBAR
icon_path = 'F:\\Niranjan Projects\\Deepfake Face Project\\image streamlit\\logo.png'
st.sidebar.image(icon_path, width=300) 
st.sidebar.title('Dashboard')
options = st.sidebar.selectbox("Select Page", ["Home", "About", "Deepfake Detection"])

# HOME PAGE
if (options == 'Home'):
    st.header("DEEPFAKE DETECTION SYSTEM")
    home_img = Image.open("F:\\Niranjan Projects\\Deepfake Face Project\\image streamlit\\deep1.jpg")
    st.image(home_img, use_column_width = True)
    st.markdown("""
    WELCOME TO THE DEEPFAKE DISEASE DETECTION SYSTEM ! 
           AI POWERED FACE AND AUDIO SWAP DETECTION
    
    In today's digital age, deepfake technology has emerged as a powerful tool capable of manipulating both visual and audio content with stunning precision.
    We harness the power of cutting-edge AI and machine learning to protect individuals and organizations from the threats posed by deepfake media. Our advanced detection algorithms analyze both facial and audio features to identify tampered content with high accuracy. Whether it's identifying swapped faces in videos or detecting manipulated audio clips, our platform ensures the authenticity of media in an era where seeing—and hearing—isn't always believing.

    Key Features:

    1.Face Swap Detection: We specialize in recognizing subtle changes in facial features and expressions, identifying any suspicious alterations in video content.
                
    2.Audio Swap Detection: Our technology detects manipulated voice clips and ensures that audio tracks haven't been swapped or modified to mislead.  
                
                """ )

elif(options == "About"):
    # Title 
    st.title("Deepfake Detection")

    video_url = 'F:\\Niranjan Projects\\Deepfake Face Project\\image streamlit\\Explanatory Video ERROR404.mp4'
    st.video(video_url)

    # Additional content
    st.write("This video demonstrates how our deepfake detection works. Basically it explains the architecture of our model")





# DEEPFAKE VIDEO DETECTOR
elif(options == 'Deepfake Detection'):
    # Load the trained ResNet50 model
    model = load_model('F:\\Niranjan Projects\\Deepfake Face Project\\model.h5')

    # Create a Streamlit app
    st.title("Deepfake Video Detector")


    st.image('F:\\Niranjan Projects\\Deepfake Face Project\\image streamlit\\deep2.jpg')

    # Step 1: User inputs the video file path
    video_path = st.file_uploader("Please upload the video file (either real or fake): ")


    if(st.button("Detect")):
        if video_path is not None:
            with st.spinner('Processing video...'):
                video_bytes = video_path.read()
                video_path = 'temp.mp4'
                with open(video_path, 'wb') as f:
                    f.write(video_bytes)
                video_frames = extract_frames(video_path)
                video_frames = preprocess_frames(video_frames)

                # Step 2: Predict the video's authenticity (real or fake)
                video_prediction = predict_deepfake(video_frames, model)

                # Step 3: Print the prediction result
                st.write(f'Prediction for the input video: {video_prediction:.2f}')

                # Step 4: Interpretation of the result (assuming a threshold of 0.5)
                threshold = 0.5
                if video_prediction < threshold:
                    st.success("The video is classified as REAL.")
                    st.markdown("""
                        <style>
                            .stApp {
                                background: linear-gradient(to right,  #00FA9A, #009900);
                                height: 100vh;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    st.write('<h1 style="color: black;">Real Video Detected</h1>', unsafe_allow_html=True)


                else:
                    st.error("The video is classified as FAKE.")
                    st.markdown("""
                        <style>
                            .stApp {
                                background: linear-gradient(to right, #ff6f61, #de1a1a);
                                height: 100vh;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    st.write('<h1 style="color: white;">Fake Video Detected</h1>', unsafe_allow_html=True)
