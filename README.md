# Real-Time Face Detection and Counter

This is a web application built with Python, OpenCV, and Streamlit that performs real-time face detection and counting using a live webcam feed.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL_HERE.streamlit.app/)

## 🚀 Features

-   **Real-Time Detection**: Detects multiple faces from a live webcam stream.
-   **Face Counting**: Displays a live count of the number of faces detected in the frame.
-   **Confidence Score**: Shows the confidence level for each detected face.
-   **Interactive UI**: Built with Streamlit for an easy-to-use web interface.
-   **Adjustable Threshold**: Allows users to adjust the detection confidence threshold in real-time.

## 🛠️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vinaykumar99999/OpenCV-Based-Real-Time-Face-Tracker-and-Counter.git](https://github.com/vinaykumar99999/OpenCV-Based-Real-Time-Face-Tracker-and-Counter.git)
    cd OpenCV-Based-Real-Time-Face-Tracker-and-Counter
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 💻 Technologies Used

-   **Python**
-   **OpenCV**: For the core computer vision functionality.
-   **Streamlit**: To create the interactive web application.
-   **streamlit-webrtc**: To handle real-time video streaming between the browser and the server.
-   **DNN Module (OpenCV)**: Using a pre-trained Caffe model for face detection.
-   **NumPy**: For numerical operations and handling image arrays.
