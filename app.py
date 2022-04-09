import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model =  keras.models.load_model('model.h5')
class_names=['Green', 'Midripen', 'Overripen', 'Yellowish_Green']

def model_predict(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    preds = tf.nn.softmax(predictions[0])
    return preds

@app.route('/', methods=['GET'])

def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #Get the file from post request
        f = request.files['file']
        
        # Save file to /uploads 
        basepath =  os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)
        
        # Make predictions
        preds = model_predict(file_path, model)
    return str("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(preds)], 100 * np.max(preds)))

if __name__ == "__main__":
    app.run(debug=False)
