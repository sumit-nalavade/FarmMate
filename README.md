# Sugarcane Disease Detection Web Application 🌱

This is a Flask-based web application that uses a trained Convolutional Neural Network (CNN) to predict sugarcane crop diseases from images and suggest appropriate fertilizer recommendations. It also integrates real-time weather data using the OpenWeatherMap API and maintains a history of all predictions in an SQLite database.

---

💡 Features

- 🌾 Predicts sugarcane leaf diseases from uploaded images.
- 🧪 Provides fertilizer recommendations based on the prediction.
- ☁️ Displays weather forecast based on user location.
- 💾 Saves prediction history locally with image, prediction, and recommendation.

---

🛠️ Tech Stack

- Python 3
- Flask
- TensorFlow / Keras
- SQLite3
- OpenWeatherMap API
- HTML/CSS (Jinja2 Templates)

---

🚀 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name


Install required dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download or place the trained model

Place your sugarcane_disease_model.h5 in the models/ directory.

Run the application

bash
Copy
Edit
python app.py
Open your browser and navigate to:
http://127.0.0.1:5000

