from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
from yolo_detection_images import runModel

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('./index.html')

@app.route("/image")
def imagepage():
    return render_template("uploadimage.html")

@app.route("/video")
def videopage():
    return render_template("uploadvideo.html")




@app.route('/detectObject' , methods=['POST'])
def mask_image():
	# print(request.files , file=sys.stderr)
	file = request.files["imagefile"].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	# img[img > 150] = 0


	img = runModel(img)

	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})







	
@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(debug = True)
