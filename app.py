from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)


#load model
model = tf.keras.models.load_model('keras_model.h5')


#define the target image size for the model
target_size = (224, 224)
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0 #normalize the image
    img_array = np.expand_dims(img_array, axis=0) #add a dimension to the image array
    return img_array


#first route
@app.route('/')
def home():
    return render_template('index.html')


#second route
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}),
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'file name is empty'})
    
    try:
        img_array = preprocess_image(file)
        #make predictions
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        if class_index == 0:
            result = 'cat'  
        else:
            result = 'dog'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})
    
    

if __name__ == '__main__':
    app.run(debug=True)
    
