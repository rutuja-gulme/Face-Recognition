# Face Data Collection using OpenCV

import cv2
import numpy as np
import os


# ---------------- Camera and Face Detection ----------------

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []

dataset_path = "./data/"
file_name = input("Enter the name of the person: ")

# Create the data directory if it does not exist
os.makedirs(dataset_path, exist_ok=True)


# ---------------- Face Data Collection ----------------

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    if len(faces) == 0:
        continue

    # Sort faces by area and select the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    for face in faces[-1:]:
        x, y, w, h = face

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        # Extract face region
        offset = 10
        face_section = frame[
            y - offset:y + h + offset,
            x - offset:x + w + offset
        ]

        face_section = cv2.resize(face_section, (100, 100))

        # Store every 10th detected face
        skip += 1

        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", face_section)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord("q"):
        break


# ---------------- Save Face Data ----------------

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

print(face_data.shape)

file_path = dataset_path + file_name + ".npy"
np.save(file_path, face_data)

print("Face data saved successfully at " + file_path)


# Release resources
cap.release()
cv2.destroyAllWindows()