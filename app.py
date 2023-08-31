from flask import Flask, render_template, request
import face_recognition
import cv2
import os
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
app = Flask(__name__, template_folder ='templates')

def run_face_recognition():
    present=[]
    # face recognition code here
    video_capture = cv2.VideoCapture(0)
    tata_image = face_recognition.load_image_file("photos/ratantata.jpg")
    tata_encoding = face_recognition.face_encodings(tata_image)[0]

    modi_image = face_recognition.load_image_file("photos/modi_1.jpg")
    modi_encoding = face_recognition.face_encodings(modi_image)[0]

    dhoni_image = face_recognition.load_image_file("photos/dhoni.jpg")
    dhoni_encoding = face_recognition.face_encodings(dhoni_image)[0]

    
    known_face_encodings = [
        tata_encoding,
        modi_encoding,
        dhoni_encoding,
        
    ]

    known_faces_names = [
        "ratantata",
        "modi",
        "dhoni",
        
    ]

    students = known_faces_names.copy()

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings , face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    present.append(name)
                    students.remove(name)
                    print(students)
                    current_time = datetime.now().strftime("%H-%M-%S")

        cv2.imshow("attendance system", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            
            video_capture.release()
            with open('attendance.csv','w+',newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        for i in range(len(present)):
                            csvwriter.writerow([present[i] , current_time])
            break
    present_count = len(present)
    absent_count = len(known_faces_names) - present_count
    students = [present_count, absent_count]
    status = ['present', 'absent']
    cl = ['red', 'green']

    plt.pie(students, labels=status, autopct='%2.1f%%', colors=cl)
    plt.title('Attendance Distribution')
    chart_image_path = 'static/pie_chart.png'  # Save in the 'static' folder
    plt.savefig(chart_image_path)


    plt.clf()
    plt.show()
@app.route("/")

def index():
    return render_template("index.html")

@app.route("/run_face_recognition", methods=["POST"])
def run_face_recognition_endpoint():
    run_face_recognition()
    return "Face Recognition completed"

@app.route("/project_details", methods=["POST"])
def project_details():
    return render_template("project.html")

@app.route("/attendance_chart",methods=["POST"])
def attendance_chart():
    return render_template("attendance_chart.html")

if __name__ == '__main__':
    app.run(debug=True, port=5001)