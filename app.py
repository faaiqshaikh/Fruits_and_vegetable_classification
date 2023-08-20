import os
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your pretrained model
model = tf.keras.models.load_model('my_model.h5')

# Get the labels from sub-directory names
labels = sorted(os.listdir('fruits_and_vegetables_classification/train'))

@app.route('/')
def index():
    return render_template('index.html', prediction='', image='')

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    if image:
        # Save the uploaded image
        filename = secure_filename(image.filename)
        image_path = os.path.join('static/images', filename)
        image.save(image_path)
        
        img = Image.open(image_path)
        img_array = np.expand_dims(np.array(img.resize((224, 224))), axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        prediction = model.predict(img_array)
        
        # Get the predicted label
        predicted_label_index = np.argmax(prediction)
        predicted_label = labels[predicted_label_index]
        
        return render_template('index.html', prediction=predicted_label, image=image_path)
    return render_template('index.html', prediction='', image='')

if __name__ == '__main__':
    app.run(debug=True)
