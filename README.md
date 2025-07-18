Real-Time Face Detection and Counter
This is a web application built with Python, OpenCV, and Streamlit that performs real-time face detection and counting using a live webcam feed.

🚀 Features
Real-Time Detection: Detects multiple faces from a live webcam stream.

Face Counting: Displays a live count of the number of faces detected in the frame.

Confidence Score: Shows the confidence level for each detected face.

Interactive UI: Built with Streamlit for an easy-to-use web interface.

Adjustable Threshold: Allows users to adjust the detection confidence threshold in real-time.

🛠️ How to Run Locally
Clone the repository:

git clone https://github.com/vinaykumar99999/OpenCV-Based-Real-Time-Face-Tracker-and-Counter.git
cd OpenCV-Based-Real-Time-Face-Tracker-and-Counter

Install the dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

💻 Technologies Used
Python

OpenCV: For the core computer vision functionality.

Streamlit: To create the interactive web application.

streamlit-webrtc: To handle real-time video streaming between the browser and the server.

DNN Module (OpenCV): Using a pre-trained Caffe model for face detection.

NumPy: For numerical operations and handling image arrays.
