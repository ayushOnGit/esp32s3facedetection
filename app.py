# import numpy as np
# import threading
# import time

# app = Flask(__name__)
# face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# # Shared frame buffer with lock
# latest_frame = None
# lock = threading.Lock()

# @app.route('/upload', methods=['POST'])
# def handle_upload():
#     global latest_frame
#     try:
#         frame = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        
#         # Face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
#         # Draw bounding boxes
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         # Update shared frame
#         with lock:
#             _, buffer = cv2.imencode('.jpg', frame)
#             global latest_frame
#             latest_frame = buffer.tobytes()
            
#         return "OK", 200
    
#     except Exception as e:
#         print(f"Processing error: {str(e)}")
#         return "Error", 500

# def generate_frames():
#     while True:
#         with lock:
#             current_frame = latest_frame
        
#         if current_frame is not None:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
#         else:
#             # Send empty frame if no data
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
#             time.sleep(0.1)

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/hello')
# def hello_world():
#     return "Hello, World!"

# if __name__ == '__main__':



# from flask import Flask, Response, request
# import cv2
# import numpy as np
# import threading
# import time
# from queue import Queue

# app = Flask(__name__)
# face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# # Shared resources with thread safety
# frame_queue = Queue(maxsize=5)  # Buffer up to 5 frames
# processed_frame = None
# lock = threading.Lock()

# def process_frames():
#     global processed_frame
#     while True:
#         if not frame_queue.empty():
#             start_time = time.time()
            
#             # Get frame data from queue
#             frame_data = frame_queue.get()
            
#             # Decode and process frame
#             frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
#             # Optimized processing pipeline
#             small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
#             gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
#             # Faster face detection parameters
#             faces = face_cascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.2,
#                 minNeighbors=3,
#                 minSize=(30, 30),
#                 flags=cv2.CASCADE_SCALE_IMAGE
#             )
            
#             # Scale coordinates back to original size
#             faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces]
            
#             # Draw bounding boxes
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
#             # Encode with optimized settings
#             _, buffer = cv2.imencode('.jpg', frame, [
#                 cv2.IMWRITE_JPEG_QUALITY, 60,
#                 cv2.IMWRITE_JPEG_OPTIMIZE, 1
#             ])
            
#             # Update processed frame
#             with lock:
#                 processed_frame = buffer.tobytes()
            
#             # Log processing performance
#             print(f"Processed frame in {(time.time()-start_time)*1000:.1f}ms")

# @app.route('/upload', methods=['POST'])
# def handle_upload():
#     try:
#         if frame_queue.full():
#             # Discard oldest frame if queue is full
#             frame_queue.get()
            
#         # Put raw frame data in queue
#         frame_queue.put(request.data)
#         return "OK", 200
    
#     except Exception as e:
#         print(f"Upload error: {str(e)}")
#         return "Error", 500

# def generate_frames():
#     while True:
#         with lock:
#             current_frame = processed_frame
        
#         if current_frame is not None:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
#         else:
#             # Send blank frame if no data
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0\x00\x10JFIF\x00' + b'\r\n')
#             time.sleep(0.01)

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     # Start processing thread
#     threading.Thread(target=process_frames, daemon=True).start()
    
#     # Enable OpenCV optimizations
#     cv2.setUseOptimized(True)
#     cv2.ocl.setUseOpenCL(True)
    
#     # Run with production server
#     from waitress import serve
#     serve(app, host="0.0.0.0", port=5999)



#working code ......................................................................................................................
# from flask import Flask, Response, request
# import cv2
# import numpy as np
# import threading
# import time
# from queue import Queue

# app = Flask(__name__)
# face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# # Shared resources with thread safety
# frame_queue = Queue(maxsize=5)  # Buffer up to 5 frames
# processed_frame = None
# lock = threading.Lock()

# def process_frames():
#     global processed_frame
#     while True:
#         if not frame_queue.empty():
#             start_time = time.time()
            
#             # Get frame data from queue
#             frame_data = frame_queue.get()
            
#             # Decode and process frame
#             frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
#             # Optimized processing pipeline
#             small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#             gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
#             # Faster face detection parameters
#             faces = face_cascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.2,
#                 minNeighbors=3,
#                 minSize=(30, 30),
#                 flags=cv2.CASCADE_SCALE_IMAGE
#             )
            
#             # Scale coordinates back to original size
#             faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]
            
#             # Draw bounding boxes
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
#             # Encode with optimized settings
#             _, buffer = cv2.imencode('.jpg', frame, [
#                 cv2.IMWRITE_JPEG_QUALITY, 60,
#                 cv2.IMWRITE_JPEG_OPTIMIZE, 1
#             ])
            
#             # Update processed frame
#             with lock:
#                 processed_frame = buffer.tobytes()
            
#             # Log processing performance
#             print(f"Processed frame in {(time.time() - start_time) * 1000:.1f}ms")

# @app.route('/')
# def hello_world():
#     return "Hello, World! 🚀 Welcome to Face Detection API!"

# @app.route('/upload', methods=['POST'])
# def handle_upload():
#     try:
#         if frame_queue.full():
#             # Discard oldest frame if queue is full
#             frame_queue.get()
        
#         # Put raw frame data in queue
#         frame_queue.put(request.data)
#         return "OK", 200
    
#     except Exception as e:
#         print(f"Upload error: {str(e)}")
#         return "Error", 500

# def generate_frames():
#     while True:
#         with lock:
#             current_frame = processed_frame
        
#         if current_frame is not None:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
#         else:
#             # Send blank frame if no data
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0\x00\x10JFIF\x00' + b'\r\n')
#             time.sleep(0.01)

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     # Start processing thread
#     threading.Thread(target=process_frames, daemon=True).start()
    
#     # Enable OpenCV optimizations
#     cv2.setUseOptimized(True)
#     cv2.ocl.setUseOpenCL(True)
    
#     # Run with production server
#     from waitress import serve
#     serve(app, host="0.0.0.0", port=5999)



from flask import Flask, Response, request
import cv2
import numpy as np
import threading
import time
from queue import Queue

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# Shared resources with thread safety
frame_queue = Queue(maxsize=5)
processed_frame = None
last_processed_time = None
current_face_status = False
lock = threading.Lock()

def process_frames():
    global processed_frame, last_processed_time, current_face_status
    while True:
        if not frame_queue.empty():
            start_time = time.time()
            
            # Get frame data from queue
            frame_data = frame_queue.get()
            
            # Decode and process frame
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Optimized processing pipeline
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection with optimized parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Update face status and timestamp with lock
            with lock:
                last_processed_time = time.time()
                current_face_status = len(faces) > 0

            # Scale coordinates back to original size
            faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]
            
            # Draw bounding boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Encode with optimized settings
            _, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 60,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            
            # Update processed frame
            with lock:
                processed_frame = buffer.tobytes()
            
            # Performance logging
            print(f"Processed frame in {(time.time() - start_time) * 1000:.1f}ms")

@app.route('/')
def hello_world():
    return "Face Detection Server 🚀 - Endpoints: /upload (POST), /video_feed, /face_status"

@app.route('/upload', methods=['POST'])
def handle_upload():
    try:
        if frame_queue.full():
            frame_queue.get()  # Discard oldest frame if queue is full
        
        frame_queue.put(request.data)
        return "OK", 200
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return "Error", 500

def generate_frames():
    while True:
        with lock:
            current_frame = processed_frame
        
        if current_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        else:
            # Send blank frame if no data available
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0\x00\x10JFIF\x00' + b'\r\n')
            time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_status')
def face_status():
    with lock:
        last_time = last_processed_time
        status = current_face_status
    
    if last_time is None:
        return Response(b'\x00', mimetype='application/octet-stream')
    
    current_time = time.time()
    if current_time - last_time <= 3.0 and status:
        return Response(b'\xD1', mimetype='application/octet-stream')
    else:
        return Response(b'\x00', mimetype='application/octet-stream')

if __name__ == '__main__':
    # Start processing thread
    threading.Thread(target=process_frames, daemon=True).start()
    
    # Enable OpenCV optimizations
    cv2.setUseOptimized(True)
    cv2.ocl.setUseOpenCL(True)
    
    # Configure production server
    from waitress import serve
    serve(app, host="0.0.0.0", port=5999)



