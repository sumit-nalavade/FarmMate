from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import os
import sqlite3
import requests  # <-- Added for Weather API

app = Flask(__name__, static_folder='web_application/static', static_url_path='/static')

# Load trained model
MODEL_PATH = os.path.join('models', 'sugarcane_disease_model.h5')
model = load_model(MODEL_PATH)

# Labels for prediction output
labels = ["healthy", "mosaic", "redrot", "rust", "yellow"]

# Fertilizer recommendations
fertilizer_recommendations = {
    "healthy": "No fertilizer needed. Crop is healthy!",
    "mosaic": "Apply balanced NPK fertilizer and control virus spread.",
    "redrot": "Use fungicide-treated fertilizers and improve soil drainage.",
    "rust": "Apply sulfur-based fertilizers and rust-specific fungicides.",
    "yellow": "Apply nitrogen-rich fertilizers to boost plant recovery."
}

# Uploads folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- DATABASE SETUP ---
DATABASE = 'history.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  image_path TEXT, 
                  prediction TEXT, 
                  fertilizer TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- WEATHER API FUNCTION ---
def get_weather(lat, lon):
    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # <-- Replace with your actual API key
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        print(f"API response: {response.status_code}")  # Log the response status code
        data = response.json()

        if response.status_code != 200:
            print(f"Error fetching weather data: {data.get('message')}")
            return {"temp": "N/A", "humidity": "N/A", "description": "N/A", "emergency": False}

        weather = {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "emergency": 'rain' in data["weather"][0]["description"].lower()  # Example emergency message
        }
        print(f"Weather data fetched: {weather}")  # Log fetched weather data
        return weather
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"temp": "N/A", "humidity": "N/A", "description": "N/A", "emergency": False}

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_weather', methods=['POST'])
def get_weather_api():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    print(f"Received geolocation: Lat = {lat}, Lon = {lon}")  # Log received coordinates
    weather = get_weather(lat, lon)
    return jsonify(weather)

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

        # Get fertilizer recommendation
        fertilizer = fertilizer_recommendations.get(result, "No recommendation available.")

        # Save image
        img_file.stream.seek(0)
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(saved_path)

        # Save to Database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('INSERT INTO history (image_path, prediction, fertilizer) VALUES (?, ?, ?)',
                  (saved_path, result, fertilizer))
        conn.commit()
        conn.close()

        return render_template('result.html', prediction=result, fertilizer=fertilizer, image_path=saved_path)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/history')
def history():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY id DESC')
    data = c.fetchall()
    conn.close()
    return render_template('history.html', history=data)

if __name__ == '__main__':
    app.run(debug=True)
