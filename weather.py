from flask import Flask, request, jsonify
import requests

# Initialize the Flask app
app = Flask(__name__)

# Replace with your actual Weather API key
API_KEY = "your_actual_api_key_here"

@app.route('/get_weather')
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    # Fetch weather data from the API
    weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    response = requests.get(weather_url)
    data = response.json()

    weather_info = {
        'temp': data['main']['temp'],
        'humidity': data['main']['humidity'],
        'description': data['weather'][0]['description'],
        'emergency': 'rain' in data['weather'][0]['description'].lower()  # Example emergency logic
    }

    return jsonify(weather_info)

# Add this line to run the app
if __name__ == '__main__':
    app.run(debug=True)
