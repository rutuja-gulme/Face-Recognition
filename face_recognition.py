# Face Recognition using K-Nearest Neighbors (KNN)

import cv2
import numpy as np
import os


# ---------------- KNN ----------------

def distance(v1, v2):
    """Compute Euclidean distance between two vectors."""
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    distances = []

    for i in range(train.shape[0]):
        # Get feature vector and label
        feature_vector = train[i, :-1]
        label = train[i, -1]

        # Compute distance from test point
        d = distance(test, feature_vector)
        distances.append([d, label])

    # Select k nearest points
    nearest = sorted(distances, key=lambda x: x[0])[:k]

    # Get labels of nearest points
    labels = np.array(nearest)[:, -1]

    # Find the most frequent label
    unique_labels, counts = np.unique(labels, return_counts=True)
    index = np.argmax(counts)

    return unique_labels[index]


# ---------------- Camera and Face Detection ----------------

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = "./data/"

face_data = []
labels = []

class_id = 0
names = {}


# ---------------- Data Preparation ----------------

for filename in os.listdir(dataset_path):
    if filename.endswith(".npy"):

        # Map class ID to person's name
        names[class_id] = filename[:-4]
        print("Loaded " + filename)

        # Load face data
        data_item = np.load(dataset_path + filename)
        face_data.append(data_item)

        # Create labels for each face sample
        target = class_id * np.ones((data_item.shape[0],))

        class_id += 1
        labels.append(target)


# Combine all face data
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

# Create the complete training dataset
trainset = np.concatenate((face_dataset, face_labels), axis=1)

print(trainset.shape)


# ---------------- Real-Time Face Recognition ----------------

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    if len(faces) == 0:
        continue

    for face in faces:
        x, y, w, h = face

        # Extract face region
        offset = 10
        face_section = frame[
            y - offset:y + h + offset,
            x - offset:x + w + offset
        ]

        face_section = cv2.resize(face_section, (100, 100))

        # Predict the face label
        output = knn(trainset, face_section.flatten())

        # Map predicted label to person's name
        predicted_name = names[int(output)]

        # Display name and bounding box
        cv2.putText(
            frame,
            predicted_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()