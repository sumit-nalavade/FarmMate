from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import os   

app = Flask(__name__,static_folder='web_application/static', static_url_path='/static')    

# Load trained model
MODEL_PATH = os.path.join('models', 'sugarcane_disease_model.h5')
model = load_model(MODEL_PATH)

# Labels for prediction output
labels = ["healthy", "mosaic", "redrot", "rust", "yellow"]

# Uploads folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    img_file = request.files['image']

    if img_file.filename == '':
        return redirect(url_for('index'))

    try:
        img = Image.open(BytesIO(img_file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        result = labels[predicted_class[0]]

        # Save image for display
        img_file.stream.seek(0)  # Reset stream position
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(saved_path)

        return render_template('result.html', prediction=result, image_path=saved_path)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
