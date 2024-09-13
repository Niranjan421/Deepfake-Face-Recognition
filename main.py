# Import necessary libraries
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

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
icon_path = 'F:\\Niranjan\\rr.png'
st.sidebar.image(icon_path, width=300) 
st.sidebar.title('Dashboard')
options = st.sidebar.selectbox("Select Page", ["Home", "About", "Deepfake Detection", "Audio Detection"])

# HOME PAGE
if (options == 'Home'):
    st.header("DEEPFAKE DETECTION SYSTEM")
    home_img = Image.open("image/deep1.jpg")
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

    video_url = 'F:\\Niranjan\\image\\Explanatory Video ERROR404.mp4'
    st.video(video_url)

    # Additional content
    st.write("This video demonstrates how our deepfake detection works. Basically it explains the architecture of our model")





# DEEPFAKE VIDEO DETECTOR
elif(options == 'Deepfake Detection'):
    # Load the trained ResNet50 model
    model = load_model('F:\\Niranjan\\model.h5')

    # Create a Streamlit app
    st.title("Deepfake Video Detector")


    st.image('F:\\Niranjan\\image\\deep2.jpg')

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

elif(options == 'Audio Detection'):
    # Load the trained model (e.g. ConvNeXt or other audio classification model)
    model = load_model('F:\\Niranjan\\audio\\FAKEDEEP\\saved_models\\audio_classification.keras')

    # Create a Streamlit app
    st.title("Audio Deepfake Detector")

    st.image('F:\\Niranjan\\1_nC8NtGCK-t0t89VrnDj4cA.png')

    # Step 1: User inputs the audio file path
    audio_path = st.file_uploader("Please upload the audio file (either real or fake): ")

    if(st.button("Detect")):
        if audio_path is not None:
            with st.spinner('Processing audio...'):
                audio_bytes = audio_path.read()
                audio_path = 'temp.wav'
                with open(audio_path, 'wb') as f:
                    f.write(audio_bytes)
                

               

                audio, sample_rate = librosa.load(filename, sr=None, res_type='kaiser_fast')

                # Extract MFCC features
                mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

                # Reshape for prediction
                mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

                # Predict the class probabilities
                predictions = model.predict(mfccs_scaled_features)

                # Get the predicted class
                predicted_label = np.argmax(predictions, axis=1)

                # Convert predicted label to class name
                prediction_class = labelencoder.inverse_transform(predicted_label)
                predicted_probabilities = model.predict(mfccs_scaled_features)

                print("MFCCs scaled features:", mfccs_scaled_features)
                print("Predicted probabilities:", predicted_probabilities)
                print("Predicted label:", predicted_label)
                print("Prediction class:", prediction_class)

                if(predicted_label==[0]):
                    print("this audio is fake")
                else:
                    print("this audio is real") 


                # Step 4: Interpretation of the result (assuming a threshold of 0.5)
                threshold = 0.5
                if predicted_label < threshold:
                    st.success("The audio is classified as REAL.")
                    st.markdown("""
                        <style>
                            .stApp {
                                background: linear-gradient(to right,  #00FA9A, #009900);
                                height: 100vh;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    st.write('<h1 style="color: black;">Real Audio Detected</h1>', unsafe_allow_html=True)

                else:
                    st.error("The audio is classified as FAKE.")
                    st.markdown("""
                        <style>
                            .stApp {
                                background: linear-gradient(to right, #ff6f61, #de1a1a);
                                height: 100vh;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    st.write('<h1 style="color: white;">Fake Audio Detected</h1>', unsafe_allow_html=True)