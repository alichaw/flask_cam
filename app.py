from flask import Flask,render_template,Response

import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])


if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

from deepface import DeepFace

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def putText(x,y,text,size=50,color=(255,255,255)):
    global img
    font = ImageFont.truetype('arial.ttf', size) 
    imgPil = Image.fromarray(img)   
    draw = ImageDraw.Draw(imgPil) 
    draw.text((x, y), text, fill=color, font=font) 
    img = np.array(imgPil)   

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        img = cv2.resize(frame,(384,240))
        try:
            analyze = DeepFace.analyze(img, actions=['emotion']) #一旦加入這行程式碼就爛掉
            #emotion = analyze['dominant_emotion']
            #putText(0, 40, emotion)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        except:
            pass
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)