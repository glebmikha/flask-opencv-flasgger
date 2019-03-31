from flask import Flask, jsonify, request, send_file, abort, redirect
from flasgger import Swagger
import cv2
from imutils import url_to_image

import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app = Flask(__name__)
Swagger(app)

@app.route('/')
def index():
    return redirect('/apidocs/')

@app.route('/facedetect',methods=['POST'])
def face_detect():
    """Detect face on image from url
    ---
    parameters:
      - name: url
        in: body
        required: true
        example: {'url':'https://vignette.wikia.nocookie.net/silicon-valley/images/b/bb/Binding_Arbitration_Image_4.jpg/revision/latest?cb=20151228222226'}
    responses:
      200:
        description: image with detected face
    """

    try:
      content = request.get_json()

      img = url_to_image(content['url'])
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)

      for (x,y,w,h) in faces:
          cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
          roigray = gray[y:y+h, x:x+w]
          roicolor = img[y:y+h, x:x+w]
      
      timestr = time.strftime("%Y%m%d-%H%M%S")
      
      fname = f'./static/file{timestr}.jpg'
      
      cv2.imwrite(fname,img)

      return send_file(fname, mimetype='image/jpg')
    
    except Exception as e:
      print(e)
      return abort(400)
