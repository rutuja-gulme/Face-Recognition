# Face Recognition using OpenCV and KNN

## Overview

A Python-based real-time face recognition system using OpenCV for face detection and a manually implemented K-Nearest Neighbors (KNN) algorithm for face classification.

The project captures face data through a webcam, stores it as NumPy arrays, and recognizes known faces in real time.

## Key Features

* Real-time face detection using Haar Cascade
* Webcam-based face data collection
* Manual KNN implementation
* Euclidean distance-based classification
* Real-time face recognition with name labels
* NumPy-based data storage

## Technologies

* Python
* OpenCV
* NumPy
* K-Nearest Neighbors (KNN)
* Haar Cascade

## How It Works

1. Capture face samples from a webcam.
2. Detect faces using Haar Cascade.
3. Resize and flatten the detected face region.
4. Store face data as `.npy` files.
5. Load stored face data for recognition.
6. Apply KNN using Euclidean distance.
7. Display the predicted name and bounding box in real time.

## Project Structure

```text
Face-Recognition/
├── face_data_collection.py
├── face_recognition.py
├── haarcascade_frontalface_alt.xml
├── README.md
├── .gitignore
└── data/                 # Generated locally and ignored by Git
```

## Setup

Install the required libraries:

```bash
pip install opencv-python numpy
```

Make sure `haarcascade_frontalface_alt.xml` is available in the project directory.

## Usage

### Step 1: Collect Face Data

```bash
python face_data_collection.py
```

Enter the person's name when prompted. The collected face data is saved in the local `data/` folder.

### Step 2: Run Face Recognition

```bash
python face_recognition.py
```

The system loads the stored face data and performs real-time face recognition through the webcam.

Press **Q** to exit.

## Notes

* Face data is generated locally and is not included in the repository.
* The `data/` folder is excluded using `.gitignore`.
* This project is a supporting Python and machine learning project demonstrating computer vision and KNN concepts.
