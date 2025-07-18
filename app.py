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
        # Convert the video frame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        h, w = img.shape[:2]

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network
        self.net.setInput(blob)
        detections = self.net.forward()

        count = 0
        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONF_THRESHOLD:
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and confidence
                text = f"{confidence * 100:.1f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 20
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the face count on the frame
        cv2.putText(img, f"Faces: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
