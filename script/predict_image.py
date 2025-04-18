import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)

# Load trained model
model = tf.keras.models.load_model(r"C:\Users\Tanmay\OneDrive\Documents\SUMIT NALAVADE\models\sugarcane_disease_model.keras")

# Class Names
train_dir = r"C:\Users\Tanmay\OneDrive\Documents\SUMIT NALAVADE\data\train"
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=1, class_mode='categorical', shuffle=False
)
class_names = list(train_generator.class_indices.keys())

# Prediction Function
def predict_disease(img_path):
    # Load and process image
    img = image.load_img(img_path, target_size=IMG_SIZE)  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]  # Get predicted class
    confidence = np.max(predictions) * 100  # Get confidence

    return predicted_class, confidence

# Test with a new image
test_image = r"C:\Users\Tanmay\OneDrive\Documents\SUMIT NALAVADE\data\val\rust\rust (84).jpeg"  # Replace with your image path
result, confidence = predict_disease(test_image)

# Print the result
print(f"🌾 Predicted Disease: {result} ({confidence:.2f}% confidence)")
