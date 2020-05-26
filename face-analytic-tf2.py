"""
Face Analytic (Expression) with TensorFlow  
  
Copyright 2020  I Made Agus Dwi Suarjaya, Kobar Septyanus, Author 3, Author 4  
  
Description     : Try to analyze faces with TensorFlow and classify into 7 expressions (angry, disgust, fear, happy, neutral, sad, surprise)  
Dataset source  : https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from flask import send_from_directory
from flask import request
from flask import Flask
import os
import ast
import time
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
app = Flask(__name__)

# ------------------------------
# Load the model and Dictionary
# ------------------------------

global sess
global graph
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
tf.compat.v1.keras.backend.set_session(sess)
model = tf.keras.models.load_model('./1590218618_model')

with open('./1590218618_dict') as dict_file:
    d = ast.literal_eval(dict_file.readline())


def predict_face(images_path):
    # --------------------------------------------
    # Predict and plot some images (From Local)
    # --------------------------------------------
    img = image.load_img(images_path, target_size=(
        48, 48), color_mode="grayscale")
    x = image.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
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
# -----------------------------------
# Flask codes for web service
# -----------------------------------
@app.route('/upload/<path:filename>', methods=['GET'])
def upload(filename):
    return send_from_directory('upload', filename)


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
                if id >= 9:
                    break
                ffn, fext = os.path.splitext(cfile.filename)
                f_path = './upload/' + str(id)
                cfile.save(f_path)
                p_result = p_result + '<th>' + predict_face(f_path) + '</th>'
                p_images = p_images + '<td><img src="' + f_path + '?' + \
                    str(time.time()) + '" style="width:100px;height:auto;"></td>'

            predict_result = ('<table style="border:1px solid black;margin-left:auto;margin-right:auto;"><tr>'
                              + p_result + '</tr><tr>' + p_images + '</tr></table>')

    return homepage_head + homepage_body + predict_result + homepage_end


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)
