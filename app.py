from flask import Flask, render_template, request, redirect, url_for
import face_recognition
import cv2
import os
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__, template_folder='templates')

# Define the run_face_recognition function
def run_face_recognition(input_file=None):
    present = []
    
    # Initialize video capture
    video_capture = cv2.VideoCapture(input_file) if input_file else cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Video capture not initialized.")
        return
    
    # Load known face encodings and names
    tata_image = face_recognition.load_image_file("photos/ratantata.jpg")
    tata_encoding = face_recognition.face_encodings(tata_image)[0]

    modi_image = face_recognition.load_image_file("photos/modi_1.jpg")
    modi_encoding = face_recognition.face_encodings(modi_image)[0]

    dhoni_image = face_recognition.load_image_file("photos/dhoni.jpg")
    dhoni_encoding = face_recognition.face_encodings(dhoni_image)[0]

    known_face_encodings = [tata_encoding, modi_encoding, dhoni_encoding]
    known_faces_names = ["ratantata", "modi", "dhoni"]

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not read frame.")
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
            face_names.append(name)
            if name:
                present.append(name)
                
        cv2.imshow("attendance system", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break
    
    # Calculate attendance statistics
    present_count = len(present)
    absent_count = len(known_faces_names) - present_count
    
    # Generate a pie chart
    students = [present_count, absent_count]
    status = ['present', 'absent']
    cl = ['green', 'red']
    plt.pie(students, labels=status, autopct='%2.1f%%', colors=cl)
    plt.title('Attendance Distribution')
    chart_image_path = 'static/pie_chart.png'  # Save in the 'static' folder
    plt.savefig(chart_image_path)
    plt.clf()
    
    # Save attendance data to a CSV file
    current_time = datetime.now().strftime("%H-%M-%S")
    with open(f'attendance_{current_time}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for student in present:
            csvwriter.writerow([student, current_time])

# Define the root route for the web application
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.form.get("photo_upload"):
            return render_template("photo_upload.html")
        elif request.form.get("video_upload"):
            return render_template("video_upload.html")
        elif request.form.get("live_image"):
            run_face_recognition()
            return redirect(url_for('index'))
    return render_template("index.html")

# Route for processing uploaded photos
@app.route("/process_photo", methods=["POST"])
def process_photo():
    if 'photo' not in request.files:
        return "No file part"
    
    photo = request.files['photo']
    
    if photo.filename == '':
        return "No selected file"
    
    if photo:
        # Save the uploaded photo to a temporary location and process it
        photo_path = "photos/dhoni.jpeg"
        photo.save(photo_path)
        run_face_recognition(photo_path)
        return redirect(url_for('index'))

# Route for processing uploaded videos
@app.route("/process_video", methods=["POST"])
def process_video():
    if 'video' not in request.files:
        return "No file part"
    
    video = request.files['video']
    
    if video.filename == '':
        return "No selected file"
    
    if video:
        # Save the uploaded video to a temporary location and process it
        video_path = "temp/video.mp4"
        video.save(video_path)
        run_face_recognition(video_path)
        return redirect(url_for('index'))
@app.route("/project_details", methods=["POST"])
def project_details():
    # Code to display project details (you need to implement this)
    return render_template("project.html")
@app.route("/run_face_recognition" , methods=["POST"])
def run_face_recognition_endpoint():
    run_face_recognition()
    return "Face recognition completed"
@app.route("/attendance_chart", methods=["POST"])
def attendance_chart():
    # Code to display attendance chart (you need to implement this)
    return render_template("attendance_chart.html")

if __name__ == "__main__":
    app.run(debug=True)
