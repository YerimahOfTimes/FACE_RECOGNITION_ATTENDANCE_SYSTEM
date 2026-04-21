import cv2
import pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine
import pandas as pd
import os
from datetime import datetime

# =========================
# LOAD FACE DATABASE
# =========================
with open("face_database.pkl", "rb") as f:
    database = pickle.load(f)

# =========================
# ATTENDANCE FILE SETUP (FIXED)
# =========================
file_path = "attendance.csv"

if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(file_path, index=False)

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# FACE RECOGNITION FUNCTION
# =========================


def recognize_face(face_img, database):
    try:
        embedding = DeepFace.represent(
            face_img,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]
    except:
        return "Error"

    best_match = None
    lowest_distance = 1

    for person, embeddings in database.items():
        for emb in embeddings:
            dist = cosine(embedding, emb)

            if dist < lowest_distance:
                lowest_distance = dist
                best_match = person

    if lowest_distance < 0.4:
        return best_match
    else:
        return "Unknown"

# =========================
# ATTENDANCE FUNCTION (SAFE VERSION)
# =========================


def mark_attendance(name):
    file_path = "attendance.csv"

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    already_marked = (
        (df["Name"] == name) &
        (df["Date"] == date)
    ).any()

    if not already_marked:
        new_row = {
            "Name": name,
            "Date": date,
            "Time": time
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(file_path, index=False)

        print(f"🧾 Attendance marked: {name}")


# =========================
# START WEBCAM
# =========================
cap = cv2.VideoCapture(0)

last_marked = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        name = recognize_face(face_img, database)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display name
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

        # =========================
        # ATTENDANCE LOGIC
        # =========================
        if name != "Unknown" and name != "Error":
            if name != last_marked:
                mark_attendance(name)
                last_marked = name

    cv2.imshow("Face Recognition Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
