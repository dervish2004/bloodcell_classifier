from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf  # ✅ <-- ADD THIS
import numpy as np
import os

app = Flask(__name__)
model = load_model("Blood_Cell.h5")
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    path = os.path.join("static", file.filename)
    file.save(path)

    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    result = classes[np.argmax(pred)]

    return render_template('result.html', result=result, image=path)

if __name__ == '__main__':
    app.run(debug=True)
