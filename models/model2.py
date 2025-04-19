import tensorflow as tf

# Load your model
model = tf.keras.models.load_model(r"C:\Users\Tanmay\OneDrive\Documents\GitHub\FarmGenious\models\sugarcane_disease_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
