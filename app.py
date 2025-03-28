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



# from flask import Flask, Response, request
# import cv2
# import numpy as np
# import threading
# import time
# from queue import Queue

# app = Flask(__name__)
# face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# # Shared resources with thread safety
# frame_queue = Queue(maxsize=5)
# processed_frame = None
# last_processed_time = None
# current_face_status = False
# lock = threading.Lock()

# def process_frames():
#     global processed_frame, last_processed_time, current_face_status
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
            
#             # Face detection with optimized parameters
#             faces = face_cascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.2,
#                 minNeighbors=3,
#                 minSize=(30, 30),
#                 flags=cv2.CASCADE_SCALE_IMAGE
#             )
            
#             # Update face status and timestamp with lock
#             with lock:
#                 last_processed_time = time.time()
#                 current_face_status = len(faces) > 0

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
            
#             # Performance logging
#             print(f"Processed frame in {(time.time() - start_time) * 1000:.1f}ms")

# @app.route('/')
# def hello_world():
#     return "Face Detection Server 🚀 - Endpoints: /upload (POST), /video_feed, /face_status"

# @app.route('/upload', methods=['POST'])
# def handle_upload():
#     try:
#         if frame_queue.full():
#             frame_queue.get()  # Discard oldest frame if queue is full
        
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
#             # Send blank frame if no data available
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0\x00\x10JFIF\x00' + b'\r\n')
#             time.sleep(0.01)

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/face_status')
# def face_status():
#     with lock:
#         last_time = last_processed_time
#         status = current_face_status
    
#     if last_time is None:
#         return Response(b'\x00', mimetype='application/octet-stream')
    
#     current_time = time.time()
#     if current_time - last_time <= 3.0 and status:
#         return Response(b'\xD1', mimetype='application/octet-stream')
#     else:
#         return Response(b'\x00', mimetype='application/octet-stream')

# if __name__ == '__main__':
#     # Start processing thread
#     threading.Thread(target=process_frames, daemon=True).start()
    
#     # Enable OpenCV optimizations
#     cv2.setUseOptimized(True)
#     cv2.ocl.setUseOpenCL(True)
    
#     # Configure production server
#     from waitress import serve
#     serve(app, host="0.0.0.0", port=5999)




# from flask import Flask, Response, request
# import cv2
# import numpy as np
# import threading
# import time
# from queue import Queue

# app = Flask(__name__)
# face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# # Shared resources with thread safety
# frame_queue = Queue(maxsize=5)
# processed_frame = None
# last_processed_time = None
# current_face_status = False
# lock = threading.Lock()

# # Movement detection variables
# movement_state = "stop"
# previous_face_data = None  # (center_x, center_y, width, height)
# movement_lock = threading.Lock()
# FRAME_CENTER_THRESHOLD = 0.2  # 20% of frame width considered center
# SIZE_CHANGE_THRESHOLD = 0.15  # 15% size change for forward/backward
# POSITION_CHANGE_THRESHOLD = 0.1  # 10% position change for left/right

# def calculate_movement(current_data, frame_width, frame_height):
#     global previous_face_data
    
#     if previous_face_data is None:
#         return "stop"
    
#     (curr_cx, curr_cy, curr_w, curr_h) = current_data
#     (prev_cx, prev_cy, prev_w, prev_h) = previous_face_data
    
#     # Calculate position changes
#     dx = curr_cx - prev_cx
#     dw = curr_w - prev_w
    
#     # Normalize changes
#     position_change = abs(dx) / frame_width
#     size_change = abs(dw) / prev_w
    
#     # Determine movement
#     if position_change > POSITION_CHANGE_THRESHOLD:
#         return "left" if dx < 0 else "right"
#     elif size_change > SIZE_CHANGE_THRESHOLD:
#         return "forward" if dw > 0 else "backward"
    
#     # Check if centered
#     frame_center_x = frame_width / 2
#     if abs(curr_cx - frame_center_x) < (frame_width * FRAME_CENTER_THRESHOLD):
#         return "stop"
    
#     return "stop"

# def process_frames():
#     global processed_frame, last_processed_time, current_face_status, previous_face_data, movement_state
#     while True:
#         if not frame_queue.empty():
#             start_time = time.time()
            
#             # Get frame data from queue
#             frame_data = frame_queue.get()
            
#             # Decode and process frame
#             frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
#             frame_height, frame_width = frame.shape[:2]
            
#             # Optimized processing pipeline
#             small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#             gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
#             # Face detection with optimized parameters
#             faces = face_cascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.2,
#                 minNeighbors=3,
#                 minSize=(30, 30),
#                 flags=cv2.CASCADE_SCALE_IMAGE
#             )
            
#             # Scale coordinates back to original size
#             faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]
            
#             current_face = None
#             if len(faces) > 0:
#                 # Track largest face
#                 current_face = max(faces, key=lambda f: f[2] * f[3])
#                 x, y, w, h = current_face
#                 center_x = x + w/2
#                 center_y = y + h/2
#                 current_data = (center_x, center_y, w, h)
                
#                 # Calculate movement
#                 if previous_face_data is not None:
#                     new_state = calculate_movement(current_data, frame_width, frame_height)
#                     with movement_lock:
#                         movement_state = new_state
                
#                 previous_face_data = current_data
#             else:
#                 with movement_lock:
#                     movement_state = "no_face"
#                 previous_face_data = None
            
#             # Draw bounding boxes
#             if current_face is not None:
#                 x, y, w, h = current_face
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
#             # Encode and update frame
#             _, buffer = cv2.imencode('.jpg', frame, [
#                 cv2.IMWRITE_JPEG_QUALITY, 60,
#                 cv2.IMWRITE_JPEG_OPTIMIZE, 1
#             ])
            
#             with lock:
#                 processed_frame = buffer.tobytes()
#                 last_processed_time = time.time()
#                 current_face_status = len(faces) > 0
            
#             print(f"Processed frame in {(time.time() - start_time) * 1000:.1f}ms")

# @app.route('/')
# def index():
#     return """
#     <h1>Face Tracking Server</h1>
#     <p>Endpoints:</p>
#     <ul>
#         <li>POST /upload - Stream video frames</li>
#         <li>GET /video_feed - Get processed video stream</li>
#         <li>GET /face_status - Check face detection status</li>
#         <li>GET /movement - Get current movement state</li>
#     </ul>
#     """

# @app.route('/upload', methods=['POST'])
# def handle_upload():
#     try:
#         if frame_queue.full():
#             frame_queue.get()
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
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0\x00\x10JFIF\x00' + b'\r\n')
#             time.sleep(0.01)

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/face_status')
# def face_status():
#     with lock:
#         last_time = last_processed_time
#         status = current_face_status
    
#     if last_time is None:
#         return Response(b'\x00', mimetype='application/octet-stream')
    
#     current_time = time.time()
#     if current_time - last_time <= 3.0 and status:
#         return Response(b'\xD1', mimetype='application/octet-stream')
#     else:
#         return Response(b'\x00', mimetype='application/octet-stream')

# @app.route('/movement')
# def get_movement():
#     with movement_lock:
#         current_state = movement_state
#     return Response(current_state, mimetype='text/plain')

# if __name__ == '__main__':
#     threading.Thread(target=process_frames, daemon=True).start()
#     cv2.setUseOptimized(True)
#     cv2.ocl.setUseOpenCL(True)
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

# Configuration
FRAME_QUEUE_SIZE = 5
PROCESSING_INTERVAL = 0.05  # 50ms between processing frames
MOVEMENT_SMOOTHING = 0.3    # Lower = more smoothing

# Movement thresholds (tune these as needed)
LEFT_THRESHOLD = -0.2       # Face center < -20% of frame width
RIGHT_THRESHOLD = 0.2       # Face center > 20% of frame width 
FORWARD_THRESHOLD = 1.25    # 25% size increase
BACKWARD_THRESHOLD = 0.8    # 20% size decrease
CENTER_ZONE = 0.15          # 15% center area

# Shared resources
frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
processed_frame = None
frame_lock = threading.Lock()

# Movement tracking
movement_state = "stop"
face_history = []  # Stores last 3 face positions for smoothing
movement_lock = threading.Lock()

def smooth_movement(current_pos):
    """Apply exponential smoothing to movement detection"""
    global face_history
    
    if len(face_history) == 0:
        face_history = [current_pos] * 3
        return current_pos
    
    # Simple moving average
    face_history.pop(0)
    face_history.append(current_pos)
    
    smoothed_x = sum(pos[0] for pos in face_history) / 3
    smoothed_size = sum(pos[1] for pos in face_history) / 3
    
    return (smoothed_x, smoothed_size)

def determine_movement(face_data, frame_size):
    """Calculate movement direction based on face position and size"""
    x, y, w, h = face_data
    frame_w, frame_h = frame_size
    
    # Normalized face center (-0.5 to 0.5)
    rel_x = ((x + w/2) / frame_w) - 0.5  
    rel_size = w * h / (frame_w * frame_h) * 10  # Scaled size
    
    # Apply smoothing
    smoothed_x, smoothed_size = smooth_movement((rel_x, rel_size))
    
    # Check movement thresholds
    if smoothed_size > FORWARD_THRESHOLD:
        return "forward"
    elif smoothed_size < BACKWARD_THRESHOLD:
        return "backward"
    elif smoothed_x < LEFT_THRESHOLD:
        return "left"
    elif smoothed_x > RIGHT_THRESHOLD:
        return "right"
    elif abs(smoothed_x) < CENTER_ZONE:
        return "stop"
    else:
        return "stop"

def process_frames():
    global processed_frame, movement_state
    
    while True:
        start_time = time.time()
        
        if not frame_queue.empty():
            # Get and decode frame
            frame_data = frame_queue.get()
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            frame_size = (frame.shape[1], frame.shape[0])
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # Track largest face
                main_face = max(faces, key=lambda f: f[2]*f[3])
                x, y, w, h = main_face
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Update movement state
                new_state = determine_movement(main_face, frame_size)
                with movement_lock:
                    movement_state = new_state
            else:
                with movement_lock:
                    movement_state = "no_face"
            
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 80,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            
            with frame_lock:
                processed_frame = buffer.tobytes()
        
        # Maintain consistent processing rate
        elapsed = time.time() - start_time
        if elapsed < PROCESSING_INTERVAL:
            time.sleep(PROCESSING_INTERVAL - elapsed)

@app.route('/movement')
def movement_endpoint():
    """Endpoint that returns current movement state"""
    with movement_lock:
        return Response(
            movement_state,
            mimetype='text/plain',
            headers={'Cache-Control': 'no-cache'}
        )

@app.route('/upload', methods=['POST'])
def upload_frame():
    """Endpoint for receiving video frames"""
    try:
        if not frame_queue.full():
            frame_queue.put(request.data)
        return "OK", 200
    except Exception as e:
        print(f"Upload error: {e}")
        return "Error", 500

def generate_feed():
    """Generator for streaming processed frames"""
    while True:
        with frame_lock:
            frame = processed_frame
        
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Endpoint for streaming processed video"""
    return Response(
        generate_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # Start processing thread
    processor = threading.Thread(target=process_frames, daemon=True)
    processor.start()
    
    # Configure and run server
    cv2.setUseOptimized(True)
    cv2.ocl.setUseOpenCL(True)
    
    from waitress import serve
    serve(
        app,
        host="0.0.0.0",
        port=5999,
        threads=4,
        channel_timeout=60
    )



