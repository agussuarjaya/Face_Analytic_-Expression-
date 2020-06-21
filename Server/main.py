# for flask environment
from datetime import datetime
import logging
import os
from flask import Flask, redirect, render_template, request
from flask_cors import CORS


# for google environment
from google.cloud import datastore
from google.cloud import storage


# for tensorflow processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
import ast
import numpy as np
import tensorflow as tf


# get bucket env
CLOUD_STORAGE_BUCKET = os.environ.get('CLOUD_STORAGE_BUCKET')


# app config
tf.compat.v1.disable_eager_execution()
app = Flask(__name__)
CORS(app, origins="*", allow_headers=[
    "Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
    supports_credentials=True, intercept_exceptions=False)


# Load the model and Dictionary
global sess
global graph
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
tf.compat.v1.keras.backend.set_session(sess)
model = tf.keras.models.load_model('./1590218618_model.h5')
with open('./1590218618_dict') as dict_file:
    d = ast.literal_eval(dict_file.readline())


# preprocess image
def preprocess_image_from_gcstorage(image_url):
    r = requests.get(image_url, stream=True)
    img = Image.open(r.raw).convert('L').resize((48, 48))
    return img


# predict image
def predict_face(img):
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


@app.route('/')
def homepage():
    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # Use the Cloud Datastore client to fetch information from Datastore about
    # each photo.
    query = datastore_client.query(kind='Expressions1')
    image_entities = list(query.fetch())

    # Return all images and expressions
    return {'all images and expressions': image_entities}


@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    photo = request.files['file']

    # Create a Cloud Storage client.
    storage_client = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(photo.filename)
    blob.upload_from_string(
        photo.read(), content_type=photo.content_type)

    # Make the blob publicly viewable.
    blob.make_public()

    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # Fetch the current date / time.
    current_datetime = datetime.now()

    # The kind for the new entity.
    kind = 'Expressions1'

    # The name/ID for the new entity.
    name = blob.name

    # Create the Cloud Datastore key for the new entity.
    key = datastore_client.key(kind, name)

    # Make prediction
    prepocessed = preprocess_image_from_gcstorage(blob.public_url)
    p_result = predict_face(prepocessed)

    # Construct the new entity using the key. Set dictionary values for entity
    # keys blob_name, storage_public_url, timestamp, and expression.
    entity = datastore.Entity(key)
    entity['blob_name'] = blob.name
    entity['image_public_url'] = blob.public_url
    entity['timestamp'] = current_datetime
    entity['expression'] = p_result

    # Save the new entity to Datastore.
    datastore_client.put(entity)

    # Use the Cloud Datastore client to fetch information from Datastore about
    # each photo.
    query = datastore_client.query(kind='Expressions1')
    image_entities = list(query.fetch())

    # Return prediction
    return {'blob_name': blob.name, 'expression': p_result, 'image_public_url':  blob.public_url, 'timestamp': current_datetime}


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
