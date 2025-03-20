import numpy as np
import threading
import time

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# Shared frame buffer with lock
latest_frame = None
lock = threading.Lock()

@app.route('/upload', methods=['POST'])
def handle_upload():
    global latest_frame
    try:
        frame = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Update shared frame
        with lock:
            _, buffer = cv2.imencode('.jpg', frame)
            global latest_frame
            latest_frame = buffer.tobytes()
            
        return "OK", 200
    
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return "Error", 500

def generate_frames():
    while True:
        with lock:
            current_frame = latest_frame
        
        if current_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        else:
            # Send empty frame if no data
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hello')
def hello_world():
    return "Hello, World!"

if __name__ == '__main__':
