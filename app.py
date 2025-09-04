from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model(MODEL_PATH)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return jsonify({'class': predicted_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
