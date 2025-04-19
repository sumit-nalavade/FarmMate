from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import os
import sqlite3
import requests
from datetime import datetime
from gtts import gTTS  # NEW IMPORT

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

# Sound folder
SOUND_FOLDER = 'static/sounds'
os.makedirs(SOUND_FOLDER, exist_ok=True)
app.config['SOUND_FOLDER'] = SOUND_FOLDER

# Database setup
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

# Weather API
def get_weather_by_location(lat, lon):
    API_KEY = "f858e2f4bf58da9aca0426006931940b"  # <-- Replace with your real API key
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        print(f"API response: {response.status_code}")
        data = response.json()

        if response.status_code != 200:
            print(f"Error fetching weather data: {data.get('message')}")
            return {"forecast": [], "emergency": False}

        forecast_data = []
        emergency = False
        seen_dates = set()

        for item in data['list']:
            forecast_date = datetime.utcfromtimestamp(item['dt']).strftime('%Y-%m-%d')
            if forecast_date not in seen_dates:
                seen_dates.add(forecast_date)
                day_forecast = {
                    "date": forecast_date,
                    "temp": item['main']['temp'],
                    "humidity": item['main']['humidity'],
                    "description": item['weather'][0]['description']
                }
                if 'rain' in item['weather'][0]["description"].lower() or item['main']['temp'] > 35:
                    emergency = True
                forecast_data.append(day_forecast)

        return {"forecast": forecast_data, "emergency": emergency}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"forecast": [], "emergency": False}

# NEW FUNCTION: generate gTTS sound
def generate_sound(text, filename):
    tts = gTTS(text=text, lang='en')
    filepath = os.path.join(app.config['SOUND_FOLDER'], filename)
    tts.save(filepath)
    return filepath

# ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_weather', methods=['POST'])
def get_weather_api():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    print(f"Received geolocation: Lat = {lat}, Lon = {lon}")
    weather = get_weather_by_location(lat, lon)
    return jsonify(weather)

@app.route('/get_market', methods=['POST'])
def get_market_api():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')

    # Mock data (later can replace by real API call)
    markets = [
        {"name": "Green Valley Mandi", "distance_km": 5.2},
        {"name": "Sunrise Farmers Market", "distance_km": 8.5},
        {"name": "AgroFresh Wholesale", "distance_km": 12.0},
    ]

    return jsonify({"markets": markets})

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

        fertilizer = fertilizer_recommendations.get(result, "No recommendation available.")

        img_file.stream.seek(0)
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(saved_path)

        # Save to database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('INSERT INTO history (image_path, prediction, fertilizer) VALUES (?, ?, ?)',
                  (saved_path, result, fertilizer))
        conn.commit()
        conn.close()

        # Create voice file
        speech_text = f"The disease prediction is {result}. Recommended action: {fertilizer}."
        sound_filename = img_file.filename.rsplit('.', 1)[0] + '.mp3'
        generate_sound(speech_text, sound_filename)

        return render_template('result.html', prediction=result, fertilizer=fertilizer, image_path=saved_path, sound_file=sound_filename)

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

@app.route('/static/sounds/<path:filename>')
def download_sound(filename):
    return send_from_directory(app.config['SOUND_FOLDER'], filename)
    
if __name__ == '__main__':
    app.run(debug=True)
