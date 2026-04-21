:

👤 Face Recognition Attendance System

A real-time AI-based attendance system that uses computer vision and deep learning to detect faces, recognize identities, and automatically log attendance into a database with a web dashboard.

🚀 Features
🎥 Real-time face detection using webcam
🧠 Face recognition using DeepFace (FaceNet model)
🧾 Automatic attendance logging (CSV-based system)
📊 Web dashboard for viewing attendance records
❌ Duplicate prevention (one entry per person per day)
💾 Persistent storage system
🛠 Tech Stack
Python 🐍
OpenCV (face detection)
DeepFace (face recognition)
Pandas (data handling)
Streamlit (dashboard UI)
Scipy (distance calculation)
📁 Project Structure
FACE_RECOGNITION_PROJECT/
│
├── recognize.py              # Main webcam recognition system
├── dashboard.py              # Streamlit dashboard
├── face_database.pkl         # Saved face embeddings
├── attendance.csv            # Attendance records
├── dataset/                  # Face images dataset (optional)
└── README.md                 # Project documentation
⚙️ How It Works
Captures live video from webcam
Detects faces using OpenCV
Converts face into embeddings using DeepFace (FaceNet)
Compares embeddings with stored database
If match found → marks attendance
Saves data into CSV file
Displays results on Streamlit dashboard
▶️ How to Run
1. Install dependencies
pip install opencv-python deepface pandas streamlit scipy
2. Run Face Recognition System
python recognize.py
3. Run Dashboard
streamlit run dashboard.py
📊 Example Output
🧾 Attendance marked: person_A

CSV format:

Name	Date	Time
person_A	2026-04-21	10:30:15
🧠 Future Improvements
Multi-face attendance tracking
Cloud database integration
Mobile app version
Email/SMS notifications
Improved face recognition model (ArcFace)
👨‍💻 Author

Built by YerimahOfTimes
