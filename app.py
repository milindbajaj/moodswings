from flask import Flask, render_template, Response
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
    
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)
