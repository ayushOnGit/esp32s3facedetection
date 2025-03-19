from flask import Flask, Response
import cv2
import numpy as np
import urllib.request

app = Flask(__name__)

# Load the Haar cascade model
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

def generate_frames():
    # ESP32-CAM stream URL
    stream_url = 'http://<ESP32_IP>/stream'  # Replace with ESP32-CAM IP
    
    while True:
        img_resp = urllib.request.urlopen(stream_url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Simple Hello World API
@app.route('/hello')
def hello_world():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5999, threaded=True)
