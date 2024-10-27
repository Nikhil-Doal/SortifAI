from flask import Flask, render_template, Response
from teachable_machine import TeachableMachine
import cv2 as cv
import os

app = Flask(__name__)

# Load the model
model = TeachableMachine(model_path=r"keras_model.h5",
                         labels_file_path=r"labels.txt")

def generate_frames():
    cap = cv.VideoCapture(0)
    image_path = "screenshot.jpg"

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Save the frame and classify it
            cv.imwrite(image_path, frame)
            result = model.classify_image(image_path)

            class_name = result["class_name"]
            class_confidence = result["class_confidence"]

            # Draw rectangle and text
            height, width, _ = frame.shape
            start_point = (50, 50)
            end_point = (width - 50, height - 50)
            color = (0, 255, 0)
            thickness = 2
            cv.rectangle(frame, start_point, end_point, color, thickness)

            text = f'{class_name} ({class_confidence:.2f})'
            font = cv.FONT_HERSHEY_SIMPLEX
            org = (50, 40)
            font_scale = 1
            font_color = (0, 255, 0)
            cv.putText(frame, text, org, font, font_scale, font_color, thickness, cv.LINE_AA)

            # Encode frame as JPEG
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of the response stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    os.remove(image_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)
