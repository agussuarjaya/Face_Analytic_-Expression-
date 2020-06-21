"""
Face Analytic (Expression) with TensorFlow  
  
Copyright 2020  I Made Agus Dwi Suarjaya, I Putu Adi Putra Setiawan, Kobar Septyanus, Ni Luh Putu Diah Putri Maheswari
  
Description     : Try to analyze faces with TensorFlow and classify into 7 expressions (angry, disgust, fear, happy, neutral, sad, surprise)  
Dataset source  : https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset
"""
import io
import ast
import time
import base64
import numpy as np
import tensorflow as tf

from flask import Flask
from flask import request
from flask import send_from_directory
from time import gmtime, strftime
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

#------------------------------
# Load the model and Dictionary
#------------------------------

global sess
global graph
sess = tf.Session()
graph = tf.get_default_graph()
tf.keras.backend.set_session(sess)
model = tf.keras.models.load_model('./1590218618_model')

with open('./1590218618_dict') as dict_file:
    d = ast.literal_eval(dict_file.readline())

def predict_face(images_path):
    #--------------------------------------------
    # Predict and plot some images (From Local)
    #--------------------------------------------
    img = image.load_img(images_path, target_size=(48, 48), color_mode="grayscale")
    x = image.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    
    with graph.as_default():
        tf.keras.backend.set_session(sess)
        prediction = model.predict(images)

    for key, value in d.items():
        if value == np.argmax(prediction[0]):
            result = str(key)
    return result

homepage_head = '''
<!DOCTYPE html>
<html>
<style>
.center {
  text-align: center;
  padding: 50px 0;
}
</style>
<body>
'''
homepage_body = '''
<div class="center">
<h1>Face Analytic (Expression) with TensorFlow</h1>

<p>Try selecting more than one image when browsing for images (max. 9 images).</p>

<form action="" method=post enctype=multipart/form-data>
  <label for="files">Select images:</label>
  <input type="file" id="files" name="files"  accept="image/*" multiple><br><br>
  <input type="submit" value="Submit">
</form>
</div>
'''
homepage_end = '''
</body>
</html>
'''
resp = {
                    'angry':'Why you angry, keep calm..',
                    'disgust':'Do you think I am digusting?',
                    'fear':'Dont be afraid, I am here with you!',
                    'happy':'Yay, I am happy too!',
                    'neutral':'You look okay..',
                    'sad':'Owh, dont be sad please..',
                    'surprise':'Why so surprised? Haha'
                    }
#-----------------------------------
# Flask codes for web service
#-----------------------------------
@app.route("/dl")
def dl():
    try:
        return send_from_directory('.', './1590218618_model.tflite', attachment_filename='model.tflite', as_attachment = True)
    except Exception as e:
        return str(e)
    
@app.route("/stream", methods=['POST'])
def stream():
    p_result = ''
    p_images = ''
    predict_result = ''
    uploaded_files = request.files.getlist("file")

    if len(uploaded_files) > 0:
        txt = str(uploaded_files[0])
        if txt.find('application/octet-stream') == -1:
            for id, cfile in enumerate(uploaded_files):
                if id >=1: break
                predict_result = resp.get(predict_face(cfile))

    return {'blob_name': str(time.time()), 'expression': predict_result, 'image_public_url':  None,
            'timestamp': strftime("%a, %d %b %Y %H:%M:%S", gmtime())}

@app.route("/", methods=['POST', 'GET'])
def home():
    p_result = ''
    p_images = ''
    predict_result = ''
    uploaded_files = request.files.getlist("files")

    if len(uploaded_files) > 0:
        txt = str(uploaded_files[0])
        if txt.find('application/octet-stream') == -1:
            for id, cfile in enumerate(uploaded_files):
                if id >=9: break
                img = image.load_img(cfile, target_size=(100, 100))
                encoded_img  = io.BytesIO()
                img.save(encoded_img, format='jpeg')
                encoded_string = base64.b64encode(encoded_img.getvalue())

                p_result = p_result + '<th>' + predict_face(cfile) + '</th>'
                p_images = p_images + '<td><img src="data:image/jpg;base64,'+ str(encoded_string.decode('utf-8'))+'"></td>'

            predict_result = ('<table style="border:1px solid black;margin-left:auto;margin-right:auto;"><tr>'
                                    + p_result + '</tr><tr>' + p_images +'</tr></table>')

    return homepage_head + homepage_body + predict_result + homepage_end

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80) #, ssl_context='adhoc')
