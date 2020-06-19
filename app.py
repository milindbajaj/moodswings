from flask import Flask, render_template, Response
import cv2
from model import FacialExpressionModel
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def home():
    return render_template('home.html')



@app.route('/video_feed')
def video_feed():
    def gen(camera):
        while True:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = FacialExpressionModel("model.json", "model_weights.h5")
    font = cv2.FONT_HERSHEY_SIMPLEX

    class VideoCamera(object):
        def __init__(self):
            self.video = cv2.VideoCapture(0)
            # self.video = cv2.VideoCapture("C:/Users/Dell/Downloads/Project/a.jpg")

        def __del__(self):
            self.video.release()

        # returns camera frames along with bounding boxes and predictions
        def get_frame(self):
            _, fr = self.video.read()
            gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.3, 5)

            for (x, y, w, h) in faces:
                fc = gray_fr[y:y + h, x:x + w]

                roi = cv2.resize(fc, (48, 48))
                pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

                cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

            _, jpeg = cv2.imencode('.jpg', fr)
            return jpeg.tobytes()
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)