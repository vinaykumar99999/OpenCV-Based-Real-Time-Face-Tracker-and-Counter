import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import numpy as np
import av

# --- Model and Configuration Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

net = load_model()

# --- RTC Configuration for Deployment ---
# This is needed for deployment on Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Main App Logic ---
st.title("Real-Time Face Detection App 👨‍👩‍👧‍👦")
st.write("This app uses a pre-trained Caffe model to detect faces in a real-time video stream.")

# Confidence threshold slider
CONF_THRESHOLD = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# --- Video Transformer Class ---
class FaceDetector(VideoTransformerBase):
    def __init__(self):
        self.net = load_model()

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        # --- OPTIMIZATION ---
        # Resize frame for faster processing and less lag
        img = cv2.resize(img, (640, 480))
        # --------------------

        h, w = img.shape[:2]

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        # ... rest of your processing code ...

        return img

# --- WebRTC Streamer ---
webrtc_ctx = webrtc_streamer(
    key="face-detection",
    video_processor_factory=FaceDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.info(f"Using a confidence threshold of {CONF_THRESHOLD}")
else:
    st.warning("Click 'START' to begin face detection.")
