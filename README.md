# Attendance System using Python with OpenCV, CSV, NumPy, Matplotlib, Face Recognition, and Flask

This project is an Attendance System developed using Python programming language and various libraries including OpenCV, CSV, NumPy, Matplotlib, Face Recognition, and Flask. The system allows you to capture images, recognize faces, and maintain attendance records in a CSV file. Additionally, a web interface is provided using Flask to view the attendance statistics visually.

## Features

- Face detection and recognition using OpenCV and Face Recognition libraries.
- Image capturing for attendance with real-time face recognition.
- CSV file to store attendance records.
- Visual representation of attendance statistics using Matplotlib.
- Web interface using Flask to interact with the attendance system.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python (3.6 or later)
- OpenCV (`opencv-python` package)
- NumPy (`numpy` package)
- Face Recognition (`face-recognition` package)
- Matplotlib (`matplotlib` package)
- Flask (`flask` package)

You can install these packages using the following command:

```bash
pip install opencv-python numpy face-recognition matplotlib flask
```

## Usage

1. Clone or download the project repository to your local machine.

2. Navigate to the project directory:

```bash
cd attendance-system-python
```

3. Run the `attendance_system.py` script to start the attendance system:

```bash
python attendance_system.py
```

4. The system will use your computer's camera to capture images and recognize faces in real-time. Press the 'q' key to stop capturing images.

5. After capturing images, the attendance records will be saved in the `attendance.csv` file.

6. To view the attendance statistics, you can run the Flask web interface. Navigate to the `web_interface` directory:

```bash
cd web_interface
```

7. Run the Flask app:

```bash
python app.py
```

8. Open your web browser and go to `http://localhost:5000` to access the attendance statistics dashboard.

## Project Structure

The project directory is organized as follows:

- `attendance_system.py`: The main script to capture images, recognize faces, and update attendance records.
- `web_interface/`: Directory containing the Flask web interface.
  - `app.py`: Flask app script to create a web interface for attendance statistics.
  - `templates/`: Directory containing HTML templates for web pages.
  - `static/`: Directory containing static assets like CSS, JS, and images.

## Contributions

Contributions to the project are welcome. If you find any issues or want to enhance the project, feel free to open an issue or submit a pull request in the project repository.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize and extend the project according to your needs. This README provides a basic overview of the Attendance System project using the specified libraries and tools.